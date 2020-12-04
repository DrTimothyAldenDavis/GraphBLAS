//------------------------------------------------------------------------------
// gb_new: create a GraphBLAS matrix with desired format and sparsity control
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_matlab.h"

GrB_Matrix gb_new               // create and empty matrix A
(
    GrB_Type type,              // type of A
    GrB_Index nrows,            // # of rows
    GrB_Index ncols,            // # of rows
    GxB_Format_Value fmt,       // requested format, if < 0 use default
    int sparsity                // sparsity control for A, 0 for default
)
{

    // create the matrix
    GrB_Matrix A = NULL ;
    OK (GrB_Matrix_new (&A, type, nrows, ncols)) ;

    // get the default format, if needed
    if (fmt < 0)
    {
        fmt = gb_default_format (nrows, ncols) ;
    }

    // set the desired format
    GxB_Format_Value fmt_current ;
    OK1 (A, GxB_Matrix_Option_get (A, GxB_FORMAT, &fmt_current)) ;
    if (fmt != fmt_current)
    {
        OK1 (A, GxB_Matrix_Option_set (A, GxB_FORMAT, fmt)) ;
    }

    // set the desired sparsity structure
    if (sparsity != 0)
    {
        int current ;
        OK1 (A, GxB_Matrix_Option_get (A, GxB_SPARSITY_CONTROL, &current)) ;
        if (current != sparsity)
        {
            OK1 (A, GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, sparsity)) ;
        }
    }

    return (A) ;
}

