//------------------------------------------------------------------------------
// GxB_Matrix_import_FullC: import a matrix in full CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Matrix_import_FullC    // import a full matrix, held by column
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create
    GrB_Index nrows,        // matrix dimension is nrows-by-ncols
    GrB_Index ncols,
    void      **Ax,         // values, size nrows*ncols
    const GrB_Descriptor desc       // descriptor for # of threads to use
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_import_FullC (&A, type, nrows, ncols, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_import_FullC") ;
    GB_IMPORT_FULL_CHECK ;

    GrB_Index anzmax ;
    bool ok = GB_Index_multiply (&anzmax, nrows, ncols) ;
    if (!ok)
    {   GB_cov[3757]++ ;
// NOT COVERED (3757):
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }

    if (anzmax > 0)
    {   GB_cov[3758]++ ;
// NOT COVERED (3758):
        GB_RETURN_IF_NULL (Ax) ;
    }

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    // allocate just the header of the matrix, not the content
    info = GB_new (A, type, nrows, ncols, GB_Ap_null, true,
        GB_FULL, GB_Global_hyper_ratio_get ( ), 0, Context) ;
    if (info != GrB_SUCCESS)
    {   GB_cov[3759]++ ;
// NOT COVERED (3759):
        // out of memory for matrix header (size O(1))
        ASSERT (*A == NULL) ;
        return (info) ;
    }

    // transplant the user's content into the matrix
    (*A)->nzmax = anzmax ;
    (*A)->magic = GB_MAGIC ;

    if (anzmax == 0)
    {   GB_cov[3760]++ ;
// NOT COVERED (3760):
        // free the user input Ax array, if it exists
        if (Ax != NULL) GB_FREE (*Ax) ;
    }
    else
    {   GB_cov[3761]++ ;
// NOT COVERED (3761):
        // transplant Ax into the matrix
        (*A)->x = (*Ax) ;
        (*Ax) = NULL ;
    }

    //--------------------------------------------------------------------------
    // import is successful
    //--------------------------------------------------------------------------

    ASSERT (*Ax == NULL) ;
    ASSERT_MATRIX_OK (*A, "A FullC imported", GB0) ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

