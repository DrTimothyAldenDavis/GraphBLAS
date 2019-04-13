//------------------------------------------------------------------------------
// GB_mex_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2018, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Returns the status of all global settings.

// TODO: add version #, 

#include "GB_mex.h"

#define USAGE "[nthreads_max threading thread_safety format hyperratio] = GB_mex_init"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    GB_WHERE (USAGE) ;
    GB_Global_user_multithreaded_set (false) ;
    GxB_init (GrB_NONBLOCKING, mxMalloc, mxCalloc, mxRealloc, mxFree, false) ;
    GB_Global_abort_function_set (GB_mx_abort) ;
    GB_Global_malloc_tracking_set (true) ;
    GxB_set (GxB_FORMAT, GxB_BY_COL) ;

    int nthreads_max ;
    GxB_get (GxB_NTHREADS, &nthreads_max) ;
    pargout [0] = mxCreateDoubleScalar (nthreads_max) ;

    GxB_Thread_Model threading ;
    GxB_get (GxB_THREADING, &threading) ;
    pargout [1] = mxCreateDoubleScalar (threading) ;

    GxB_Thread_Model thread_safety ;
    GxB_get (GxB_THREAD_SAFETY, &thread_safety) ;
    pargout [2] = mxCreateDoubleScalar (thread_safety) ;

    GxB_Format_Value format ;
    GxB_get (GxB_FORMAT, &format) ;
    pargout [3] = mxCreateDoubleScalar (format) ;

    double hyperratio ;
    GxB_get (GxB_HYPER, &hyperratio) ;
    pargout [4] = mxCreateDoubleScalar (hyperratio) ;

    GrB_finalize ( ) ;
}

