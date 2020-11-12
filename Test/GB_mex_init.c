//------------------------------------------------------------------------------
// GB_mex_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns the status of all global settings.

#include "GB_mex.h"

#define USAGE "[nthreads_max threading thread_safety format hyper_switch" \
"name version date about license compiledate compiletime api api_about" \
" chunk] = GB_mex_init"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    GxB_init (GrB_NONBLOCKING, mxMalloc, mxCalloc, mxRealloc, mxFree, false) ;
    GB_Global_abort_function_set (GB_mx_abort) ;
    GB_Global_malloc_tracking_set (true) ;

    // MATLAB default is by column
    GxB_Global_Option_set_(GxB_FORMAT, GxB_BY_COL) ;

    int nthreads_max ;
    GxB_Global_Option_get_(GxB_NTHREADS, &nthreads_max) ;
    pargout [0] = mxCreateDoubleScalar (nthreads_max) ;

    pargout [1] = mxCreateDoubleScalar (1) ;
    pargout [2] = mxCreateDoubleScalar (1) ;

    GxB_Format_Value format ;
    GxB_Global_Option_get_(GxB_FORMAT, &format) ;
    pargout [3] = mxCreateDoubleScalar (format) ;

    double hyper_switch ;
    GxB_Global_Option_get_(GxB_HYPER_SWITCH, &hyper_switch) ;
    pargout [4] = mxCreateDoubleScalar (hyper_switch) ;

    char *name ;
    GxB_Global_Option_get_(GxB_LIBRARY_NAME, &name) ;
    pargout [5] = mxCreateString (name) ;

    int version [3] ;
    GxB_Global_Option_get_(GxB_LIBRARY_VERSION, version) ;
    pargout [6] = mxCreateDoubleMatrix (1, 3, mxREAL) ;
    double *p = mxGetPr (pargout [6]) ;
    p [0] = version [0] ;
    p [1] = version [1] ;
    p [2] = version [2] ;

    char *date ;
    GxB_Global_Option_get_(GxB_LIBRARY_DATE, &date) ;
    pargout [7] = mxCreateString (date) ;

    char *about ;
    GxB_Global_Option_get_(GxB_LIBRARY_ABOUT, &about) ;
    pargout [8] = mxCreateString (about) ;

    char *license ;
    GxB_Global_Option_get_(GxB_LIBRARY_LICENSE, &license) ;
    pargout [9] = mxCreateString (license) ;

    char *compile_date ;
    GxB_Global_Option_get_(GxB_LIBRARY_COMPILE_DATE, &compile_date) ;
    pargout [10] = mxCreateString (compile_date) ;

    char *compile_time ;
    GxB_Global_Option_get_(GxB_LIBRARY_COMPILE_TIME, &compile_time) ;
    pargout [11] = mxCreateString (compile_time) ;

    int api [3] ;
    GxB_Global_Option_get_(GxB_API_VERSION, api) ;
    pargout [12] = mxCreateDoubleMatrix (1, 3, mxREAL) ;
    double *a = mxGetPr (pargout [12]) ;
    a [0] = api [0] ;
    a [1] = api [1] ;
    a [2] = api [2] ;

    char *api_about ;
    GxB_Global_Option_get_(GxB_API_ABOUT, &api_about) ;
    pargout [13] = mxCreateString (api_about) ;

    double chunk ;
    GxB_Global_Option_get_(GxB_CHUNK, &chunk) ;
    pargout [14] = mxCreateDoubleScalar (chunk) ;

    bool use_mkl ;
    GxB_Global_Option_get_(GxB_MKL, &use_mkl) ;
    pargout [15] = mxCreateLogicalScalar (use_mkl) ;

    GrB_finalize ( ) ;
}

