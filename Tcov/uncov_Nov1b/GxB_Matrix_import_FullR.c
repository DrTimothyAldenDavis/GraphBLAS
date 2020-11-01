//------------------------------------------------------------------------------
// GxB_Matrix_import_FullR: import a matrix in full format, held by row
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_import_FullR  // import a full matrix, held by row
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    void **Ax,          // values, size nrows*ncols entries
    const GrB_Descriptor desc
)
{   GB_cov[4592]++ ;
// NOT COVERED (4592):

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_import_FullR (&A, type, nrows, ncols, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_import_FullR") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (A, type, ncols, nrows, 0, 0, false, 0,
        NULL, NULL, NULL, NULL, Ax, GxB_FULL, false, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

