//------------------------------------------------------------------------------
// GxB_Matrix_import_HyperCSC: import a matrix in hypersparse CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_import_HyperCSC      // import a hypersparse CSC matrix
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    GrB_Index nzmax,    // size of Ai and Ax
    bool jumbled,       // if true, indices in each column may be unsorted
    int64_t ignore,     // TODO::remove
    GrB_Index nvec,     // size of Ah
    GrB_Index **Ap,     // column "pointers", size nvec+1
    GrB_Index **Ah,     // columns that appear in A, size nvec
    GrB_Index **Ai,     // row indices, size nzmax
    void **Ax,          // values, size nzmax entries
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_import_HyperCSC (&A, type, nrows, ncols, nzmax,"
        " jumbled, nvec, &Ap, &Ah, &Ai, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_import_HyperCSC") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (A, type, nrows, ncols, nzmax, 0, jumbled, nvec,
        Ap, Ah, NULL, Ai, Ax, GxB_HYPERSPARSE, true, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

