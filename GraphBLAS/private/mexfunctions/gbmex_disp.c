//------------------------------------------------------------------------------
// gbmex_disp: display a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_disp (C, level)

#define FREE_WORK GrB_Matrix_free (&C_to_free) ;

#include "gb_interface.h"

#define USAGE "usage: gbmex_disp (ghb, C, level)"

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
    int arena = GrB_DEFAULT ;

    GBMX_USAGE (nargin == 3 && nargout == 0, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    arena = ghb ? GrB_DEFAULT : MXARENA ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;
    int level = (int) mxGetScalar (pargin [2]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&C, &C_to_free, &(Matrix [0]), arena, err)) ;

    //--------------------------------------------------------------------------
    // print the GraphBLAS matrix
    //--------------------------------------------------------------------------

    // print 1-based indices
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_PRINT_1BASED)) ;

    // print sizes of shallow components
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;

    char *kind ;
    switch (Matrix [0].kind)
    { 
        case KIND_GHB :     kind = "@GhB matrix" ; break ;
        case KIND_GRB :     kind = "@GrB matrix" ; break ;
        default : 
        case KIND_BUILTIN : 
            #ifdef OCTAVE
            kind = "Octave matrix" ;
            #else
            kind = "MATLAB matrix" ;
            #endif
    }

    OK (GxB_Matrix_fprint (C, kind, level, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    gb_wrapup ( ) ;
}

