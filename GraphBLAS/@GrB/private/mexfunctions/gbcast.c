//------------------------------------------------------------------------------
// gbcast: convert to a sparse or full built-in MATLAB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// MATLAB sparse or full matrix.  The output is a @GrB matrix but with a data
// structure that matches a standard built-in sparse or full matrix: full if
// all entries are present, and sparse otherwise.

// Usage:

// C = gbcast (X, type)

#define FREE_WORK                   \
    GrB_Matrix_free (&X_shallow) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = gbcast (X, type)"

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

    GrB_Matrix *C_opaque = NULL, X = NULL, X_shallow = NULL, C = NULL ;

    gbmx_usage (nargin == 2 && nargout <= 1, USAGE) ;
    pargout [0] = gbmx_export_struct (&C_opaque) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    char type_string [LEN+2] ;
    gbmx_mxstring_to_string (type_string, LEN, pargin [1], "type") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&X, &X_shallow, &(Matrix [0]))) ;

    //--------------------------------------------------------------------------
    // make a deep copy and typecast to the desired type
    //--------------------------------------------------------------------------

    GrB_Type type = gb_string_to_type (type_string) ;
    OK (gb_typecast (&C, X, type, GxB_BY_COL, GxB_SPARSE + GxB_FULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, KIND_BUILTIN)) ;   // fixme:
    gb_wrapup ( ) ;
}

