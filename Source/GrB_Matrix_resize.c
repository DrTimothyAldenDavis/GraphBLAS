//------------------------------------------------------------------------------
// GrB_Matrix_resize: change the size of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Matrix_resize      // change the size of a matrix
(
    GrB_Matrix C,               // matrix to modify
    GrB_Index nrows_new,        // new number of rows in matrix
    GrB_Index ncols_new         // new number of columns in matrix
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_resize (C, nrows_new, ncols_new)") ;
    GB_BURBLE_START ("GrB_Matrix_resize") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;

    //--------------------------------------------------------------------------
    // resize the matrix
    //--------------------------------------------------------------------------

    GrB_Info info = GB_resize (C, nrows_new, ncols_new, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

