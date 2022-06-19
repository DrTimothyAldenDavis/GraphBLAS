//------------------------------------------------------------------------------
// gbbandwidth: compute the upper and lower bandwidth of a GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// usage:

// [lo,hi] = gbbandwidth (A)

#include "gb_interface.h"

#define USAGE "usage: [lo,hi] = gbbandwidth (A)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 1 && nargout == 2, USAGE) ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // compute lo and hi
    //--------------------------------------------------------------------------

    int64_t hi = 0, lo = 0 ;
    GrB_Matrix x = NULL, imin = NULL, imax = NULL, idiag = NULL ;

    // FUTURE: this is efficient is A is held by column.  If A is held
    // by row then x=true(ncols,1), imin=A*x, ... should be done instead.

    // x = true (1, nrows)
    OK (GrB_Matrix_new (&x, GrB_BOOL, 1, nrows)) ;
    OK (GrB_Matrix_assign_BOOL (x, NULL, NULL, true, GrB_ALL, 1, GrB_ALL, nrows,
        NULL)) ;

    // imin = x*A, where imin(j) = min column index in column j
    OK (GrB_Matrix_new (&imin, GrB_INT64, 1, ncols)) ;
    OK (GrB_mxm (imin, NULL, NULL, GxB_MIN_FIRSTJ_INT64, x, A, NULL)) ;

    // imax = x*A, where imax(j) = max column index in column j
    OK (GrB_Matrix_new (&imax, GrB_INT64, 1, ncols)) ;
    OK (GrB_mxm (imax, NULL, NULL, GxB_MAX_FIRSTJ_INT64, x, A, NULL)) ;
    OK (GrB_Matrix_free (&x)) ;

    // construct idiag: idiag (j) = j with same sparsity pattern as imin
    OK (GrB_Matrix_new (&idiag, GrB_INT64, 1, ncols)) ;
    OK (GrB_Matrix_apply_IndexOp_INT64 (idiag, NULL, NULL, GrB_COLINDEX_INT64,
        imin, 0, NULL)) ;

    // imin = idiag - imin
    OK (GrB_Matrix_eWiseMult_BinaryOp (imin, NULL, NULL, GrB_MINUS_INT64,
        idiag, imin, NULL)) ;

    // imax = imax - idiag
    OK (GrB_Matrix_eWiseMult_BinaryOp (imax, NULL, NULL, GrB_MINUS_INT64,
        imax, idiag, NULL)) ;
    OK (GrB_Matrix_free (&idiag)) ;

    // hi = max (imin, 0) ;
    OK (GrB_Matrix_reduce_INT64 (&hi, GrB_MAX_INT64, GrB_MAX_MONOID_INT64,
        imin, NULL)) ;
    OK (GrB_Matrix_free (&imin)) ;

    // lo = max (imax, 0) ;
    OK (GrB_Matrix_reduce_INT64 (&lo, GrB_MAX_INT64, GrB_MAX_MONOID_INT64,
        imax, NULL)) ;
    OK (GrB_Matrix_free (&imax)) ;

    //--------------------------------------------------------------------------
    // return result as int64 scalars
    //--------------------------------------------------------------------------

    if (lo > FLINTMAX || hi > FLINTMAX)
    { 
        // output is int64 to avoid flint overflow
        int64_t *p ;
        pargout [0] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        // use mxGetData (best for Octave, fine for MATLAB)
        p = (int64_t *) mxGetData (pargout [0]) ;
        p [0] = (int64_t) lo ;
        pargout [1] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        p = (int64_t *) mxGetData (pargout [1]) ;
        p [0] = (int64_t) hi ;
    }
    else
    { 
        // output is double
        pargout [0] = mxCreateDoubleScalar ((double) lo) ;
        pargout [1] = mxCreateDoubleScalar ((double) hi) ;
    }
}

