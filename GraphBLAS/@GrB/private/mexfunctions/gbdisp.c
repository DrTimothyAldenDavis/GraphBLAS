//------------------------------------------------------------------------------
// gbdisp: display a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbdisp (C, cnz, level)

#define FREE_WORK GrB_Matrix_free (&C_shallow) ;

#include "gb_interface.h"

#define USAGE "usage: gbdisp (C, cnz, level)"

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

    GrB_Matrix C = NULL, C_shallow = NULL ;

    gbmx_usage (nargin == 3 && nargout == 0, USAGE) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    double cnz = mxGetScalar (pargin [1]) ;
    int level = (int) mxGetScalar (pargin [2]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&C, &C_shallow, &(Matrix [0]))) ;

    //--------------------------------------------------------------------------
    // print the GraphBLAS matrix
    //--------------------------------------------------------------------------

    // construct the matrix name
    char s [LEN+2] ;
    if (cnz == 0)
    { 
        snprintf (s, LEN, "no nonzeros") ;
    }
    else if (cnz == 1)
    { 
        snprintf (s, LEN, "1 nonzero") ;
    }
    else if (cnz < INT64_MAX)
    { 
        snprintf (s, LEN, GBd " nonzeros", (int64_t) cnz) ;
    }
    else
    { 
        snprintf (s, LEN, "%g nonzeros", cnz) ;
    }
    s [LEN] = '\0' ;

    // print 1-based indices
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_PRINT_1BASED)) ;

    // print sizes of shallow components
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;

    OK (GxB_Matrix_fprint (C, s, level, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    gb_wrapup ( ) ;
}

