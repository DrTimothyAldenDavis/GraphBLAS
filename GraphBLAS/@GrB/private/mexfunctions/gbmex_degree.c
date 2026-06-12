//------------------------------------------------------------------------------
// gbmex_degree: number of entries in each vector of a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// sparse matrix.

//  gbmex_degree (A, 'row')     row degree
//  gbmex_degree (A, 'col')     column degree

#define FREE_WORK                   \
    GrB_Matrix_free (&x) ;          \
    GrB_Matrix_free (&A_to_free) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&d) ;

#include "gb_interface.h"

#define USAGE "usage: degree = gbmex_degree (A, dim)"

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

    GrB_Matrix *d_opaque = NULL, d = NULL, x = NULL, A = NULL,
        A_to_free = NULL ;

    GBMX_USAGE (nargin == 2 && nargout <= 1, USAGE) ;

    pargout [0] = gbmx_export_struct (&d_opaque) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    char dim_string [LEN+2] ;
    gbmx_mxstring_to_string (dim_string, LEN, pargin [1], "dim") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), err)) ;

    uint64_t nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // compute the row/column degree
    //--------------------------------------------------------------------------

    if (MATCH (dim_string, "row"))
    { 

        //----------------------------------------------------------------------
        // row degree
        //----------------------------------------------------------------------

        // x = ones (ncols,1) ;
        OK (GrB_Matrix_new (&x, GrB_INT64, ncols, 1)) ;
        OK (GrB_Matrix_assign_INT64 (x, NULL, NULL, 1, GrB_ALL, ncols,
            GrB_ALL, 1, NULL)) ;
        // d = A*x using the PLUS_PAIR semiring
        OK (GrB_Matrix_new (&d, GrB_INT64, nrows, 1)) ;
        OK (GrB_mxm (d, NULL, NULL, GxB_PLUS_PAIR_INT64, A, x, NULL)) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // column degree
        //----------------------------------------------------------------------

        // x = ones (nrows,1) ;
        OK (GrB_Matrix_new (&x, GrB_INT64, nrows, 1)) ;
        OK (GrB_Matrix_assign_INT64 (x, NULL, NULL, 1, GrB_ALL, nrows,
            GrB_ALL, 1, NULL)) ;
        // d = A'*x using the PLUS_PAIR semiring
        OK (GrB_Matrix_new (&d, GrB_INT64, ncols, 1)) ;
        OK (GrB_mxm (d, NULL, NULL, GxB_PLUS_PAIR_INT64, A, x, GrB_DESC_T0)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (d_opaque, (GrB_Matrix *) &d, KIND_GRB, err)) ;
    gb_wrapup ( ) ;
}

