//------------------------------------------------------------------------------
// Source/Template/GB_critical_section: execute code in a critical section
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This critical section is only used to protect the global queue of matrices
// with pending operations, for GrB_wait ( ).

// Critical sections for Windows threads and ANSI C11 threads are listed below
// as drafts, but these threading models are not yet supported.

{

    //--------------------------------------------------------------------------
    // POSIX pthreads
    //--------------------------------------------------------------------------

    #if defined (USER_POSIX_THREADS)
    {
        ok = (pthread_mutex_lock (&GB_sync) == 0) ;
        GB_CRITICAL_SECTION ;
        ok = ok && (pthread_mutex_unlock (&GB_sync) == 0) ;
    }

    //--------------------------------------------------------------------------
    // Microsoft Windows
    //--------------------------------------------------------------------------

//  This should work, per the Windows spec, but is not yet supported.
//  #elif defined (USER_WINDOWS_THREADS)
//  {
//      // This should work, per the Windows spec, but is not yet supported.
//      EnterCriticalSection (&GB_sync) ;
//      GB_CRITICAL_SECTION ;
//      LeaveCriticalSection (&GB_sync) ;
//  }

    //--------------------------------------------------------------------------
    // ANSI C11 threads
    //--------------------------------------------------------------------------

//  This should work per the ANSI C11 Spec, but is not yet supported.
//  #elif defined (USER_ANSI_THREADS)
//  {
//      ok = (mtx_lock (&GB_sync) == thrd_success) ;
//      GB_CRITICAL_SECTION ;
//      ok = ok && (mtx_unlock (&GB_sync) == thrd_success) ;
//  }

    //--------------------------------------------------------------------------
    // OpenMP
    //--------------------------------------------------------------------------

    #else   // USER_OPENMP_THREADS or USER_NO_THREADS
    { 
        // default: use a named OpenMP critical section.  If OpenMP is not
        // available, then the #pragma is ignored and this becomes vanilla,
        // single-threaded code.
        #pragma omp critical(GB_critical_section)
        GB_CRITICAL_SECTION ;
    }
    #endif
}

#undef GB_CRITICAL_SECTION

