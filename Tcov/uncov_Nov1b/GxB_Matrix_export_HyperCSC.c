//------------------------------------------------------------------------------
// GxB_Matrix_export_HyperCSC: export a matrix in hypersparse CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_export_HyperCSC  // export and free a hypersparse CSC matrix
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    GrB_Index *nzmax,   // size of Ai and Ax
    bool *jumbled,      // if true, indices in each column may be unsorted
    int64_t *ignore,    // TODO::remove
    GrB_Index *nvec,    // size of Ah
    GrB_Index **Ap,     // column "pointers", size nvec+1
    GrB_Index **Ah,     // column indices, size nvec
    GrB_Index **Ai,     // row indices, size nzmax
    void **Ax,          // values, size nzmax entries
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_export_HyperCSC (&A, &type, &nrows, &ncols, &nzmax,"
        " &jumbled, &nvec, &Ap, &Ah, &Ai, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_export_HyperCSC") ;
    GB_RETURN_IF_NULL (A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    if (jumbled == NULL)
    {   GB_cov[4579]++ ;
// NOT COVERED (4579):
GB_GOTCHA ;
        // the exported matrix cannot be jumbled
        GB_MATRIX_WAIT (*A) ;
    }
    else
    {   GB_cov[4580]++ ;
// covered (4580): 5518
        // the exported matrix is allowed to be jumbled
        GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (*A) ;
    }

    //--------------------------------------------------------------------------
    // ensure the matrix is hypersparse CSC
    //--------------------------------------------------------------------------

    // ensure the matrix is in CSC format
    if (!((*A)->is_csc))
    {   GB_cov[4581]++ ;
// covered (4581): 3562
        // A = A', done in-place, to put A in CSC format
        GBURBLE ("(transpose) ") ;
        GB_OK (GB_transpose (NULL, NULL, true, *A,
            NULL, NULL, NULL, false, Context)) ;
    }

    GB_OK (GB_convert_any_to_hyper (*A, Context)) ;
    ASSERT (GB_IS_HYPERSPARSE (*A)) ;

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    info = GB_export (A, type, nrows, ncols, nzmax, NULL, jumbled, nvec,
        Ap, Ah, NULL, Ai, Ax, NULL, NULL, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

