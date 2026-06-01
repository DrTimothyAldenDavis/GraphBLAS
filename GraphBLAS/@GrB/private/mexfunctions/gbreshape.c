//------------------------------------------------------------------------------
// gbreshape: reshape a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:

// C = gbreshape (A, nrows_new, ncols_new, by_col)

#define FREE_WORK                   \
    GrB_Matrix_free (&A_to_free) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = gbreshape (A, nrows_new, ncols_new, by_col)"

#include "gb_interface.h"

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

    GrB_Matrix *C_opaque = NULL, C = NULL, A = NULL, A_to_free = NULL ;

    GBMX_USAGE ((nargin == 3 || nargin == 4) && nargout == 1, USAGE) ;

    pargout [0] = gbmx_export_struct (&C_opaque) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    uint64_t nrows_new = gbmx_get_uint64_scalar (pargin [1], "nrows_new") ;
    uint64_t ncols_new = gbmx_get_uint64_scalar (pargin [2], "ncols_new") ;
    bool by_col = (nargin == 3) ? true : ((bool) mxGetScalar (pargin [3])) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), err)) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // reshape the matrix
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_reshapeDup (&C, A, by_col, nrows_new, ncols_new, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, KIND_GRB, err)) ;
    gb_wrapup ( ) ;
}

