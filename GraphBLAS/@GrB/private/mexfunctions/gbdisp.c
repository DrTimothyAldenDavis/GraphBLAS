//------------------------------------------------------------------------------
// gbdisp: display a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbdisp (C, level)

#define FREE_WORK GrB_Matrix_free (&C_to_free) ;

#include "gb_interface.h"

#define USAGE "usage: gbdisp (C, level)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs (no outputs to construct)
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, C_to_free = NULL ;

    GBMX_USAGE (nargin == 2 && nargout == 0, USAGE) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;
    int level = (int) mxGetScalar (pargin [1]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&C, &C_to_free, &(Matrix [0]), err)) ;

    //--------------------------------------------------------------------------
    // print the GraphBLAS matrix
    //--------------------------------------------------------------------------

    // print 1-based indices
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_PRINT_1BASED)) ;

    // print sizes of shallow components
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;

    OK (GxB_Matrix_fprint (C, NULL, level, NULL)) ;
    printf ("\n") ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    gb_wrapup ( ) ;
}

