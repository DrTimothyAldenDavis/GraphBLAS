//------------------------------------------------------------------------------
// GxB_Matrix_import_CSR: import a matrix in CSR format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_import_CSR      // import a CSR matrix
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    GrB_Index nzmax,    // size of Ai and Ax
    bool jumbled,       // if true, indices in each row may be unsorted
    int64_t ignore,     // TODO::remove
    GrB_Index **Ap,     // row "pointers", size nrows+1
    GrB_Index **Aj,     // column indices, size nzmax
    void **Ax,          // values, size nzmax entries
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_import_CSR (&A, type, nrows, ncols, nzmax,"
        " jumbled, &Ap, &Aj, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_import_CSR") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (A, type, ncols, nrows, nzmax, 0, jumbled, 0,
        Ap, NULL, NULL, Aj, Ax, GxB_SPARSE, false, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

