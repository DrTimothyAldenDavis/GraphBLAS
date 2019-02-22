//------------------------------------------------------------------------------
// GB_thread_local_access: get pointer to thread-local storage
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This implementation is complete for user threading with POSIX threads,
// OpenMP, and no user threads.  Windows and ANSI C11 threads are not yet
// supported.

// not parallel: this function does O(1) work and is already thread-safe.

#include "GB.h"

//------------------------------------------------------------------------------
// GB_thread_local_access
//------------------------------------------------------------------------------

char *GB_thread_local_access ( )    // return pointer to thread-local storage
{ 

    //--------------------------------------------------------------------------
    // get pointer to thread-local-storage
    //--------------------------------------------------------------------------

    #if defined (USER_POSIX_THREADS)
    {
        if (GB_Global.user_multithreaded)
        {
            // thread-local storage for POSIX
            char *p = pthread_getspecific (GB_thread_local_key) ;
            bool ok = true ;
            if (p == NULL)
            {
                // first time:  allocate the space for the report
                p = (void *) GB_Global.calloc_function ((GB_RLEN+1), sizeof (char));
                ok = (p != NULL) ;
                ok = ok && (pthread_setspecific (GB_thread_local_key, p) == 0) ;
            }
            // do not attempt to recover from a failure to allocate the space
            return (p) ;
        }
    }
    #elif defined (USER_WINDOWS_THREADS)
    {
        // for user applications that use Windows threads:
        #error "Windows threads not yet supported"
        return (NULL) ;
    }
    #endif

    // USER_OPENMP_THREADS, USER_NO_THREADS, USER_ANSI_THREADS,
    // or USER_POSIX_THREADS but with GB_Global.user_multithreaded false.
    return (GB_thread_local_report) ;
}

