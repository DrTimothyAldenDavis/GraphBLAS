//------------------------------------------------------------------------------
// GB_mex_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2018, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Returns the # of threads that GraphBLAS can use internally.

#include "GB_mex.h"

#define USAGE "nthreads_max = GB_mex_init"

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

    GrB_finalize ( ) ;
}

