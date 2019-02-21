//------------------------------------------------------------------------------
// GB_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GrB_init (or GxB_init) must called before any other GraphBLAS operation;
// both rely on this internal function.

// GrB_finalize must be called as the last GraphBLAS operation.

// GrB_init or GxB_init define the mode that GraphBLAS will use:  blocking or
// non-blocking.  With blocking mode, all operations finish before returning to
// the user application.  With non-blocking mode, operations can be left
// pending, and are computed only when needed.

// GxB_init is the same as GrB_init except that it also defines the
// malloc/calloc/realloc/free functions to use.

// not parallel: this function does O(1) work and is already thread-safe.

#include "GB.h"

//------------------------------------------------------------------------------
// Thread local storage
//------------------------------------------------------------------------------

// Thread local storage is used to to record the details of the last error
// encountered for GrB_error.  If the user application is multi-threaded, each
// thread that calls GraphBLAS needs its own private copy of this report.

#if defined (USER_POSIX_THREADS)
// thread-local storage for POSIX THREADS
pthread_key_t GB_thread_local_report ;

#elif defined (USER_WINDOWS_THREADS)
// for user applications that use Windows threads:
#error Windows threading not yet supported

#elif defined (USER_ANSI_THREADS)
// for user applications that use ANSI C11 threads:
// (this should work per the ANSI C11 specification but is not yet supported)
_Thread_local char GB_thread_local_report [GB_RLEN+1] = "" ;

#else // USER_OPENMP_THREADS, or USER_NO_THREADS
// OpenMP user threads, or no user threads: this is the default
#pragma omp threadprivate (GB_thread_local_report)
char GB_thread_local_report [GB_RLEN+1] = "" ;
#endif

//------------------------------------------------------------------------------
// All Global storage is declared and initialized here
//------------------------------------------------------------------------------

// If the user creates threads that work on GraphBLAS matrices, then all of
// those threads must share the same matrix queue, and the same mode.

GB_Global_struct GB_Global =
{

    // queued matrices with work to do
    .queue_head = NULL,         // pointer to first queued matrix

    // GraphBLAS mode
    .mode = GrB_NONBLOCKING,    // default is nonblocking

    // initialization flag
    .GrB_init_called = false,   // GrB_init has not yet been called

    // max number of threads
    .nthreads_max = 1,          // max number of threads

    // default format
    .hyper_ratio = GB_HYPER_DEFAULT,
    .is_csc = (GB_FORMAT_DEFAULT != GxB_BY_ROW),    // default is GxB_BY_ROW

    // Sauna workspace for Gustavson's method (one per thread)
    .Saunas [0] = NULL,
    .Sauna_in_use [0] = false,

    // abort function for debugging only
    .abort_function   = abort,

    // malloc/calloc/realloc/free functions: default to ANSI C11 functions
    .malloc_function  = malloc,
    .calloc_function  = calloc,
    .realloc_function = realloc,
    .free_function    = free

    #ifdef GB_MALLOC_TRACKING
    // malloc tracking, for testing, statistics, and debugging only
    , .nmalloc = 0                // memory block counter
    , .malloc_debug = false       // do not test memory handling
    , .malloc_debug_count = 0     // counter for testing memory handling
    , .inuse = 0                  // memory space current in use
    , .maxused = 0                // high water memory usage
    #endif

} ;

//------------------------------------------------------------------------------
// GB_init
//------------------------------------------------------------------------------

// If GraphBLAS is used by multiple user threads, only one can call GrB_init
// or GxB_init.

GrB_Info GB_init            // start up GraphBLAS
(
    const GrB_Mode mode,    // blocking or non-blocking mode

    // pointers to memory management functions.  Must be non-NULL.
    void * (* malloc_function  ) (size_t),
    void * (* calloc_function  ) (size_t, size_t),
    void * (* realloc_function ) (void *, size_t),
    void   (* free_function    ) (void *),

    GB_Context Context      // from GrB_init or GxB_init
)
{

    //--------------------------------------------------------------------------
    // establish malloc/calloc/realloc/free
    //--------------------------------------------------------------------------

    // GrB_init passes in the ANSI C11 malloc/calloc/realloc/free

    GB_Global.malloc_function  = malloc_function  ;
    GB_Global.calloc_function  = calloc_function  ;
    GB_Global.realloc_function = realloc_function ;
    GB_Global.free_function    = free_function    ;

    //--------------------------------------------------------------------------
    // max number of threads
    //--------------------------------------------------------------------------

    // Maximum number of threads for internal parallelization.
    // SuiteSparse:GraphBLAS requires OpenMP to use parallelization within
    // calls to GraphBLAS.  The user application may also call GraphBLAS in
    // parallel, from multiple user threads.  The user threads can use OpenMP,
    // or POSIX pthreads.

    #if defined ( _OPENMP )
    GB_Global.nthreads_max = omp_get_max_threads ( ) ;
    #else
    GB_Global.nthreads_max = 1 ;
    #endif

    //--------------------------------------------------------------------------
    // create the global queue and thread-local storage
    //--------------------------------------------------------------------------

    GB_CRITICAL (GB_queue_create ( )) ;

    //--------------------------------------------------------------------------
    // initialize the global queue
    //--------------------------------------------------------------------------

    // Only one thread should initialize these settings.  If multiple threads
    // call GrB_init, only the first thread does this work.

    if (! (mode == GrB_BLOCKING || mode == GrB_NONBLOCKING))
    { 
        // mode is invalid; also report the error for GrB_error.
        return (GB_ERROR (GrB_INVALID_VALUE, (GB_LOG,
            "Unknown mode: %d; must be %d (GrB_NONBLOCKING) or %d"
            " (GrB_BLOCKING)", (int) mode, (int) GrB_NONBLOCKING,
            (int) GrB_BLOCKING))) ;
    }

    bool I_was_first ;

    GB_CRITICAL (GB_queue_init (mode, &I_was_first)) ;

    if (! I_was_first)
    { 
        return (GB_ERROR (GrB_INVALID_VALUE, (GB_LOG,
            "GrB_init must not be called twice"))) ;
    }

    //--------------------------------------------------------------------------
    // clear Sauna workspaces
    //--------------------------------------------------------------------------

    for (int t = 0 ; t < GxB_NTHREADS_MAX ; t++)
    {
        GB_Global.Saunas [t] = NULL ;
        GB_Global.Sauna_in_use [t] = false ;
    }

    //--------------------------------------------------------------------------
    // set the global default format
    //--------------------------------------------------------------------------

    // set the default hypersparsity ratio and CSR/CSC format;  any thread
    // can do this later as well, so there is no race condition danger.

    GB_Global.hyper_ratio = GB_HYPER_DEFAULT ;
    GB_Global.is_csc = (GB_FORMAT_DEFAULT != GxB_BY_ROW) ;

    //--------------------------------------------------------------------------
    // initialize malloc tracking (testing and debugging only)
    //--------------------------------------------------------------------------

    #ifdef GB_MALLOC_TRACKING
    {
        GB_Global.nmalloc = 0 ;
        GB_Global.malloc_debug = false ;
        GB_Global.malloc_debug_count = 0 ;
        GB_Global.inuse = 0 ;
        GB_Global.maxused = 0 ;
    }
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

