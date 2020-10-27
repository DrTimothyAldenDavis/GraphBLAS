//------------------------------------------------------------------------------
// GxB_Matrix_export_HyperCSR: export a matrix in hypersparse CSR format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_export_HyperCSR  // export and free a hypersparse CSR matrix
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    GrB_Index *nzmax,   // size of Aj and Ax
    bool *jumbled,      // if true, indices in each row may be unsorted
    int64_t *nonempty,  // number of rows with at least one entry
    GrB_Index *nvec,    // size of Ah
    GrB_Index **Ap,     // row "pointers", size nvec+1
    GrB_Index **Ah,     // row indices, size nvec
    GrB_Index **Aj,     // column indices, size nzmax
    void **Ax,          // values, size nzmax entries
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_export_HyperCSR (&A, &type, &nrows, &ncols, &nzmax,"
        " &jumbled, &nonempty, &nvec, &Ap, &Ah, &Aj, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_export_HyperCSR") ;
    GB_RETURN_IF_NULL (A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    if (jumbled == NULL)
    { 
        // the exported matrix cannot be jumbled
        GB_MATRIX_WAIT (*A) ;
    }
    else
    {
        // the exported matrix is allowed to be jumbled
        GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (*A) ;
    }

    //--------------------------------------------------------------------------
    // ensure the matrix is hypersparse CSR
    //--------------------------------------------------------------------------

    // ensure the matrix is in CSR format
    if ((*A)->is_csc)
    {
        // A = A', done in-place, to put A in CSR format
        GBURBLE ("(transpose) ") ;
        GB_OK (GB_transpose (NULL, NULL, false, *A,
            NULL, NULL, NULL, false, Context)) ;
    }

    GB_OK (GB_convert_any_to_hyper (*A, Context)) ;
    ASSERT (GB_IS_HYPERSPARSE (*A)) ;

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    info = GB_export (A, type, ncols, nrows,
        nzmax, NULL, jumbled, nonempty, nvec,
        Ap, Ah, NULL, Aj, Ax, NULL, NULL, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

