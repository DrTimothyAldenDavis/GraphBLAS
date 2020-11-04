//------------------------------------------------------------------------------
// GxB_Matrix_export_CSC: export a matrix in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_export_CSC  // export and free a CSC matrix
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    GrB_Index *nzmax,   // size of Ai and Ax
    bool *jumbled,      // if true, indices in each column may be unsorted
    int64_t *ignore,    // TODO::remove
    GrB_Index **Ap,     // column "pointers", size ncols+1
    GrB_Index **Ai,     // row indices, size nzmax
    void **Ax,          // values, size nzmax entries
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_export_CSC (&A, &type, &nrows, &ncols, &nzmax,"
        " &jumbled, &Ap, &Ai, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_export_CSC") ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;
    ASSERT_MATRIX_OK (*A, "A to export as CSC", GB0) ;

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
    // ensure the matrix is sparse CSC
    //--------------------------------------------------------------------------

    // ensure the matrix is in CSC format
    if (!((*A)->is_csc))
    { 
        // A = A', done in-place, to put A in CSC format
        GBURBLE ("(transpose) ") ;
        GB_OK (GB_transpose (NULL, NULL, true, *A,
            NULL, NULL, NULL, false, Context)) ;
    }

    GB_OK (GB_convert_any_to_sparse (*A, Context)) ;
    ASSERT (GB_IS_SPARSE (*A)) ;

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    int sparsity ;
    bool is_csc ;
    info = GB_export (A, type, nrows, ncols, nzmax, NULL, jumbled, NULL,
        Ap, NULL, NULL, Ai, Ax, &sparsity, &is_csc, Context) ;
    if (info == GrB_SUCCESS)
    {
        ASSERT (sparsity == GxB_SPARSE) ;
        ASSERT (is_csc) ;
    }
    GB_BURBLE_END ;
    return (info) ;
}

