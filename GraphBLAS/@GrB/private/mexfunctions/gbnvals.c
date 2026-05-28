//------------------------------------------------------------------------------
// gbnvals: number of entries in a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// sparse matrix.

// Usage

// nvals = gbnvals (A)

#define FREE_WORK GrB_Matrix_free (&A_shallow) ;

#include "gb_interface.h"

#define USAGE "usage: nvals = gbnvals (A)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, A_shallow = NULL ;

    gbmx_usage (nargin == 1 && nargout <= 1, USAGE) ;

    pargout [0] = mxCreateDoubleScalar (0) ;
    double *anvals_output = (double *) mxGetData (pargout [0]) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_shallow, &(Matrix [0]))) ;

    //--------------------------------------------------------------------------
    // get the # of entries in the matrix
    //--------------------------------------------------------------------------

    uint64_t nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;

    double anvals ;
    if (nvals == INT64_MAX)
    { 
        uint64_t nrows, ncols ;
        OK (GrB_Matrix_nrows (&nrows, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;
        anvals = ((double) nrows) * ((double) ncols) ;
    }
    else
    { 
        anvals = (double) nvals ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    (*anvals_output) = anvals ;
    gb_wrapup ( ) ;
}

