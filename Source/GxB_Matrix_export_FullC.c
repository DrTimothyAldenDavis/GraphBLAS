//------------------------------------------------------------------------------
// GxB_Matrix_export_FullC: export a full matrix, held by column
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_export_FullC  // export and free a full matrix, by column
(
    GrB_Matrix *A,          // handle of matrix to export and free
    GrB_Type *type,         // type of matrix exported
    GrB_Index *nrows,       // matrix dimension is nrows-by-ncols
    GrB_Index *ncols,
    void      **Ax,         // values, size nrows*ncols
    const GrB_Descriptor desc       // descriptor for # of threads to use
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_export_FullC (&A, &type, &nrows, &ncols, &Ax,"
        " desc)") ;
    GB_BURBLE_START ("GxB_Matrix_export_FullC") ;
    GB_EXPORT_FULL_CHECK ;

    GB_RETURN_IF_NULL (Ax) ;

    if (!GB_is_dense (*A))
    { 
        // A must be dense or full
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    // ensure the matrix is full
    if (!GB_IS_FULL (*A))
    { 
        // convert A from dense to full
        GB_sparse_to_full (*A) ;
    }

    // ensure the matrix is in full CSC format
    (*A)->hyper_ratio = GB_NEVER_HYPER ;
    if (!((*A)->is_csc))
    { 
        // A = A', done in place, to put A in CSC format
        GBBURBLE ("(transpose) ") ;
        GB_OK (GB_transpose (NULL, NULL, true, (*A),
            NULL, NULL, NULL, false, Context)) ;
    }

    ASSERT_MATRIX_OK ((*A), "A export: full CSC", GB0) ;
    ASSERT ((*A)->is_csc) ;

    // export the content and remove it from A
    if (nrows > 0 && ncols > 0)
    { 
        (*Ax) = (*A)->x ;
        (*A)->x = NULL ;
    }
    else
    { 
        (*Ax) = NULL ;
    }

    //--------------------------------------------------------------------------
    // export is successful
    //--------------------------------------------------------------------------

    // free the matrix header; do not free the exported content of the matrix,
    // which has already been removed above.
    GB_Matrix_free (A) ;
    ASSERT (*A == NULL) ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

