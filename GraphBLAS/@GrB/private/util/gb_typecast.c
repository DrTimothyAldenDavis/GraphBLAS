//------------------------------------------------------------------------------
// gb_typecast: typecast a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_matlab.h"

GrB_Matrix gb_typecast          // A = (atype) S, where A is deep
(
    GrB_Type atype,             // if NULL, copy but do not typecast
    GrB_Matrix S,               // may be shallow
    GxB_Format_Value fmt,       // also convert to the requested format
    int sparsity                // sparsity control for A, if 0 use S
)
{

    //--------------------------------------------------------------------------
    // determine the sparsity control for A
    //--------------------------------------------------------------------------

    sparsity = gb_get_sparsity (S, NULL, sparsity) ;

    //--------------------------------------------------------------------------
    // get the type of A and S
    //--------------------------------------------------------------------------

    GrB_Type stype ;
    OK (GxB_Matrix_type (&stype, S)) ;
    if (atype == NULL)
    { 
        // keep the same type
        atype = stype ;
    }

    //--------------------------------------------------------------------------
    // create the empty A matrix and set its format and sparsity
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, S)) ;
    OK (GrB_Matrix_ncols (&ncols, S)) ;
    GrB_Matrix A = gb_new (atype, nrows, ncols, fmt, sparsity) ;

    //--------------------------------------------------------------------------
    // A = S
    //--------------------------------------------------------------------------

    if (gb_is_integer (atype) && gb_is_float (stype))
    { 
        // A = (atype) round (S), using MATLAB rules for typecasting.
        OK1 (A, GrB_Matrix_apply (A, NULL, NULL, gb_round_binop (stype), S,
            NULL)) ;
    }
    else
    { 
        // A = (atype) S, with GraphBLAS typecasting if needed.
        OK1 (A, GrB_assign (A, NULL, NULL, S, GrB_ALL, nrows, GrB_ALL, ncols,
            NULL)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (A) ;
}

