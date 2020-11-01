//------------------------------------------------------------------------------
// GB_export: export a matrix or vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// No conversion is done, and the matrix is exported in its current sparsity
// structure and by-row/by-col format.   A->nvec_nonempty is computed if
// negative and A is sparse or hypersparse.

#include "GB_export.h"

GrB_Info GB_export      // export a matrix in any format
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix to export
    GrB_Index *vlen,    // vector length
    GrB_Index *vdim,    // vector dimension
    GrB_Index *nzmax,   // size of Ab, Ai, and Ax
    GrB_Index *nvals,   // # of entries for bitmap matrices
    bool *jumbled,      // if true, sparse/hypersparse may be jumbled
    GrB_Index *nvec,    // size of Ah for hypersparse
    GrB_Index **Ap,     // pointers, size nvec+1 for hyper, vdim+1 for sparse
    GrB_Index **Ah,     // vector indices, size nvec for hyper
    int8_t **Ab,        // bitmap, size nzmax
    GrB_Index **Ai,     // indices, size nzmax
    void **Ax,          // values, size nzmax
    int *sparsity,      // hypersparse, sparse, bitmap, or full
    bool *is_csc,       // if true then export matrix by-column, else by-row
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*A) ;
    ASSERT_MATRIX_OK (*A, "A to export", GB0) ;
    ASSERT (!GB_ZOMBIES (*A)) ;
    ASSERT (GB_JUMBLED_OK (*A)) ;
    ASSERT (!GB_PENDING (*A)) ;
    GB_RETURN_IF_NULL (type) ;
    GB_RETURN_IF_NULL (vlen) ;
    GB_RETURN_IF_NULL (vdim) ;
    GB_RETURN_IF_NULL (Ax) ;

    int s = GB_sparsity (*A) ;
    switch (s)
    {
        case GxB_HYPERSPARSE : 
            GB_RETURN_IF_NULL (nvec) ;
            GB_RETURN_IF_NULL (Ah) ;

        case GxB_SPARSE : 
            GB_RETURN_IF_NULL (nzmax) ;
            GB_RETURN_IF_NULL (Ap) ;
            GB_RETURN_IF_NULL (Ai) ;
            break ;

        case GxB_BITMAP : 
            GB_RETURN_IF_NULL (nvals) ;
            GB_RETURN_IF_NULL (Ab) ;

        case GxB_FULL : 
        default: ;
    }

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    (*type) = (*A)->type ;
    (*vlen) = (*A)->vlen ;
    (*vdim) = (*A)->vdim ;

    switch (s)
    {
        case GxB_HYPERSPARSE : 
            (*nvec) = (*A)->nvec ;
            (*Ah) = (*A)->h ; (*A)->h = NULL ;

        case GxB_SPARSE : 
            (*nzmax) = (*A)->nzmax ;
            if (jumbled != NULL)
            { 
                (*jumbled) = (*A)->jumbled ;
            }
            (*Ap) = (*A)->p ; (*A)->p = NULL ;
            (*Ai) = (*A)->i ; (*A)->i = NULL ;
            break ;

        case GxB_BITMAP : 
            (*nvals) = (*A)->nvals ;
            (*Ab) = (*A)->b ; (*A)->b = NULL ;

        case GxB_FULL : 
        default: ;
    }

    (*Ax) = (*A)->x ; (*A)->x = NULL ;

    if (sparsity != NULL)
    { 
GB_GOTCHA ;
        (*sparsity) = s ;
    }
    if (is_csc != NULL)
    { 
GB_GOTCHA ;
        (*is_csc) = (*A)->is_csc ;
    }

    GB_Matrix_free (A) ;
    ASSERT ((*A) == NULL) ;
    return (GrB_SUCCESS) ;
}

