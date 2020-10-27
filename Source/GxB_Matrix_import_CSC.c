//------------------------------------------------------------------------------
// GxB_Matrix_import_CSC: import a matrix in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_import_CSC      // import a CSC matrix
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    GrB_Index nzmax,    // size of Ai and Ax
    bool jumbled,       // if true, indices in each column may be unsorted
    int64_t nonempty,   // number of columns with at least one entry:
                        // either < 0 if not known, or >= 0 if exact
    GrB_Index **Ap,     // column "pointers", size ncols+1
    GrB_Index **Ai,     // row indices, size nzmax
    void **Ax,          // values, size nzmax entries
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_import_CSC (&A, type, nrows, ncols, nzmax,"
        " jumbled, nonempty, &Ap, &Ai, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_import_CSC") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (A, type, nrows, ncols, nzmax, 0, jumbled, nonempty, 0,
        Ap, NULL, NULL, Ai, Ax, GxB_SPARSE, true, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

