//------------------------------------------------------------------------------
// GxB_Matrix_import_HyperCSR: import a matrix in hypersparse CSR format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_import_HyperCSR      // import a hypersparse CSR matrix
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index nrows,    // number of rows of the matrix
    GrB_Index ncols,    // number of columns of the matrix
    GrB_Index nzmax,    // size of Aj and Ax
    bool jumbled,       // if true, indices in each row may be unsorted
    int64_t nonempty,   // number of rows with at least one entry:
                        // either < 0 if not known, or >= 0 if exact
    GrB_Index nvec,     // size of Ah
    GrB_Index **Ap,     // row "pointers", size nvec+1
    GrB_Index **Ah,     // rows that appear in A, size nvec
    GrB_Index **Aj,     // column indices, size nzmax
    void **Ax,          // values, size nzmax
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_import_HyperCSR (&A, type, nrows, ncols, nzmax,"
        " jumbled, nonempty, nvec, &Ap, &Ah, &Aj, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_import_HyperCSR") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    info = GB_import (A, type, ncols, nrows, nzmax, 0, jumbled, nonempty, nvec,
        Ap, Ah, NULL, Aj, Ax, GxB_HYPERSPARSE, false, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

