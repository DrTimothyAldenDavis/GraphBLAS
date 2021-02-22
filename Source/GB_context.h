//------------------------------------------------------------------------------
// GB_context.h: definitions for the internal context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CONTEXT_H
#define GB_CONTEXT_H

//------------------------------------------------------------------------------
// GB_context: error logging, thread control, and Werk space
//------------------------------------------------------------------------------

// Error messages are logged in Context->logger_handle, on the stack which is
// handle to the input/output matrix/vector (typically C).  If the user-defined
// data types, operators, etc have really long names, the error messages are
// safely truncated (via snprintf).  This is intentional, but gcc with
// -Wformat-truncation will print a warning (see pragmas above).  Ignore the
// warning.

// The Context also contains the number of threads to use in the operation.  It
// is normally determined from the user's descriptor, with a default of
// nthreads_max = GxB_DEFAULT (that is, zero).  The default rule is to let
// GraphBLAS determine the number of threads automatically by selecting a
// number of threads between 1 and nthreads_max.  GrB_init initializes
// nthreads_max to omp_get_max_threads.  Both the global value and the value in
// a descriptor can set/queried by GxB_set / GxB_get.

// Some GrB_Matrix and GrB_Vector methods do not take a descriptor, however
// (GrB_*_dup, _build, _exportTuples, _clear, _nvals, _wait, and GxB_*_resize).
// For those methods the default rule is always used (nthreads_max =
// GxB_DEFAULT), which then relies on the global nthreads_max.

// GB_WERK_SIZE is the size of a small fixed-sized array in the Context, used
// for small werkspace allocations.  GB_WERK_SIZE must be a multiple of 8.
// The Werk array is placed first in the GB_Context struct, to ensure
// proper alignment.

#define GB_WERK_SIZE 65536

typedef struct
{
    GB_void Werk [GB_WERK_SIZE] ;   // werkspace stack
    double chunk ;                  // chunk size for small problems
    const char *where ;             // GraphBLAS function where error occurred
    char **logger_handle ;          // error report
    int nthreads_max ;              // max # of threads to use
    int pwerk ;                     // top of Werk stack, initially zero
}
GB_Context_struct ;

typedef GB_Context_struct *GB_Context ;

// GB_WHERE keeps track of the currently running user-callable function.
// User-callable functions in this implementation are written so that they do
// not call other unrelated user-callable functions (except for GrB_*free).
// Related user-callable functions can call each other since they all report
// the same type-generic name.  Internal functions can be called by many
// different user-callable functions, directly or indirectly.  It would not be
// helpful to report the name of an internal function that flagged an error
// condition.  Thus, each time a user-callable function is entered (except
// GrB_*free), it logs the name of the function with the GB_WHERE macro.
// GrB_*free does not encounter error conditions so it doesn't need to be
// logged by the GB_WHERE macro.

#define GB_CONTEXT(where_string)                                    \
    /* construct the Context */                                     \
    GB_Context_struct Context_struct ;                              \
    GB_Context Context = &Context_struct ;                          \
    /* set Context->where so GrB_error can report it if needed */   \
    Context->where = where_string ;                                 \
    /* get the default max # of threads and default chunk size */   \
    Context->nthreads_max = GB_Global_nthreads_max_get ( ) ;        \
    Context->chunk = GB_Global_chunk_get ( ) ;                      \
    /* get the pointer to where any error will be logged */         \
    Context->logger_handle = NULL ;                                 \
    /* initialize the Werk stack */                                 \
    Context->pwerk = 0 ;

#define GB_WHERE(C,where_string)                                    \
    if (!GB_Global_GrB_init_called_get ( ))                         \
    {                                                               \
        return (GrB_PANIC) ; /* GrB_init not called */              \
    }                                                               \
    GB_CONTEXT (where_string)                                       \
    if (C != NULL)                                                  \
    {                                                               \
        /* free any prior error logged in the object */             \
        GB_FREE (C->logger) ;                                       \
        Context->logger_handle = &(C->logger) ;                     \
    }

#define GB_WHERE1(where_string)                                     \
    if (!GB_Global_GrB_init_called_get ( ))                         \
    {                                                               \
        return (GrB_PANIC) ; /* GrB_init not called */              \
    }                                                               \
    GB_CONTEXT (where_string)

//------------------------------------------------------------------------------
// GB_GET_NTHREADS_MAX:  determine max # of threads for OpenMP parallelism.
//------------------------------------------------------------------------------

//      GB_GET_NTHREADS_MAX obtains the max # of threads to use and the chunk
//      size from the Context.  If Context is NULL then a single thread *must*
//      be used.  If Context->nthreads_max is <= GxB_DEFAULT, then select
//      automatically: between 1 and nthreads_max, depending on the problem
//      size.  Below is the default rule.  Any function can use its own rule
//      instead, based on Context, chunk, nthreads_max, and the problem size.
//      No rule can exceed nthreads_max.

#define GB_GET_NTHREADS_MAX(nthreads_max,chunk,Context)                     \
    int nthreads_max = (Context == NULL) ? 1 : Context->nthreads_max ;      \
    if (nthreads_max <= GxB_DEFAULT)                                        \
    {                                                                       \
        nthreads_max = GB_Global_nthreads_max_get ( ) ;                     \
    }                                                                       \
    double chunk = (Context == NULL) ? GxB_DEFAULT : Context->chunk ;       \
    if (chunk <= GxB_DEFAULT)                                               \
    {                                                                       \
        chunk = GB_Global_chunk_get ( ) ;                                   \
    }

//------------------------------------------------------------------------------
// GB_nthreads: determine # of threads to use for a parallel loop or region
//------------------------------------------------------------------------------

// If work < 2*chunk, then only one thread is used.
// else if work < 3*chunk, then two threads are used, and so on.

static inline int GB_nthreads   // return # of threads to use
(
    double work,                // total work to do
    double chunk,               // give each thread at least this much work
    int nthreads_max            // max # of threads to use
)
{
    work  = GB_IMAX (work, 1) ;
    chunk = GB_IMAX (chunk, 1) ;
    int64_t nthreads = (int64_t) floor (work / chunk) ;
    nthreads = GB_IMIN (nthreads, nthreads_max) ;
    nthreads = GB_IMAX (nthreads, 1) ;
    return ((int) nthreads) ;
}

//------------------------------------------------------------------------------
// error logging
//------------------------------------------------------------------------------

// The GB_ERROR macro logs an error in the logger error string.
//
//  if (i >= nrows)
//  {
//      GB_ERROR (GrB_INDEX_OUT_OF_BOUNDS,
//          "Row index %d out of bounds; must be < %d", i, nrows) ;
//  }
//
// The user can then do:
//
//  const char *error ;
//  GrB_error (&error, A) ;
//  printf ("%s", error) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
const char *GB_status_code (GrB_Info info) ;

// maximum size of the error logger string
#define GB_LOGGER_LEN 384

// log an error in the error logger string and return the error
#define GB_ERROR(info,format,...)                                           \
{                                                                           \
    if (Context != NULL)                                                    \
    {                                                                       \
        char **logger_handle = Context->logger_handle ;                     \
        if (logger_handle != NULL)                                          \
        {                                                                   \
            (*logger_handle) = GB_MALLOC (GB_LOGGER_LEN+1, char) ;          \
            if ((*logger_handle) != NULL)                                   \
            {                                                               \
                snprintf ((*logger_handle), GB_LOGGER_LEN,                  \
                    "GraphBLAS error: %s\nfunction: %s\n" format,           \
                    GB_status_code (info), Context->where, __VA_ARGS__) ;   \
            }                                                               \
        }                                                                   \
    }                                                                       \
    return (info) ;                                                         \
}

//------------------------------------------------------------------------------
// GB_werk_push/pop: manage werkspace in the Context->Werk stack
//------------------------------------------------------------------------------

// Context->Werk is a small fixed-size array that is allocated on the stack
// of any user-callable GraphBLAS function.  It is used for small werkspace
// allocations.

// GB_ROUND8(s) rounds up s to a multiple of 8
#define GB_ROUND8(s) (((s) + 7) & (~0x7))

//------------------------------------------------------------------------------
// GB_werk_push: allocate werkspace from the Werk stack or malloc/calloc
//------------------------------------------------------------------------------

// The werkspace is allocated from the Werk static if it small enough and space
// is available.  Otherwise it is allocated by malloc or calloc.

static inline void *GB_werk_push    // return pointer to newly allocated space
(
    // output
    bool *on_stack,                 // true if werkspace is from Werk stack
    // input
    bool do_calloc,                 // if true, zero the space
    size_t nitems,                  // # of items to allocate
    size_t size_of_item,            // size of each item
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // determine where to allocate the werkspace
    //--------------------------------------------------------------------------

    size_t size ;
    if (Context == NULL || nitems > GB_WERK_SIZE || size_of_item > GB_WERK_SIZE)
    {
        // no context, or werkspace is too large to allocate from the Werk stack
        (*on_stack) = false ;
    }
    else
    {
        // try to allocate from the Werk stack
        size = GB_ROUND8 (nitems * size_of_item) ;
        ASSERT (size % 8 == 0) ;        // size is rounded up to a multiple of 8
        size_t freespace = GB_WERK_SIZE - Context->pwerk ;
        ASSERT (freespace % 8 == 0) ;   // thus freespace is also multiple of 8
        (*on_stack) = (size <= freespace) ;
    }

    //--------------------------------------------------------------------------
    // allocate the werkspace
    //--------------------------------------------------------------------------

    if (*on_stack)
    {
        // allocate the werkspace from the Werk stack
        GB_void *p = Context->Werk + Context->pwerk ;
        Context->pwerk += size ;
        if (do_calloc) memset (p, 0, size) ;
        return ((void *) p) ;
    }
    else
    {
        // allocate the werkspace from malloc/calloc
        return (do_calloc ?
            GB_calloc_memory (nitems, size_of_item) :
            GB_malloc_memory (nitems, size_of_item)) ;
    }
}

#define GB_WERK_DECLARE(X,type)                                         \
    type *GB_RESTRICT X = NULL ;                                        \
    bool X ## _on_stack = false ;                                       \
    size_t X ## _nitems = 0 ;

#define GB_WERK_PUSH(X,do_calloc,nitems,type)                           \
    X ## _nitems = (nitems) ;                                           \
    X = (type *) GB_werk_push (&(X ## _on_stack), do_calloc,            \
        X ## _nitems, sizeof (type), Context) ; 

//------------------------------------------------------------------------------
// GB_werk_pop:  free werkspace from the Werk stack
//------------------------------------------------------------------------------

// If the werkspace was allocated from the Werk stack, it must be at the top of
// the stack to free it properly.  Freeing a werkspace in the middle of the
// Werk stack also frees everything above it.  This is not a problem if that
// space is also being freed, but the assertion below ensures that the freeing
// werkspace from the Werk stack is done in LIFO order, like a stack.

static inline void *GB_werk_pop     // free the top block of werkspace memory
(
    // input/output
    void *p,                        // werkspace to free
    // input
    bool on_stack,                  // true if werkspace is from Werk stack
    size_t nitems,                  // # of items to allocate
    size_t size_of_item,            // size of each item
    GB_Context Context
)
{

    if (p == NULL)
    {
        // nothing to do
    }
    else if (on_stack)
    {
        // werkspace was allocated from the Werk stack
        size_t size = GB_ROUND8 (nitems * size_of_item) ;
        ASSERT (Context != NULL) ;
        ASSERT (size % 8 == 0) ;
        ASSERT (((GB_void *) p) + size == Context->Werk + Context->pwerk) ;
        Context->pwerk = ((GB_void *) p) - Context->Werk ;
    }
    else
    {
        // werkspace was allocated from malloc/calloc
        GB_free_memory (p) ;
    }
    return (NULL) ;                 // return NULL to indicate p was freed
}

#define GB_WERK_POP(X,type)                                         \
    X = (type *) GB_werk_pop (X, X ## _on_stack, X ## _nitems,      \
        sizeof (type), Context) ;

#endif

