//------------------------------------------------------------------------------
// GB_import: import a matrix in any format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// TODO: import shallow for MATLAB

#include "GB_export.h"

GrB_Info GB_import      // import a matrix in any format
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index vlen,     // vector length
    GrB_Index vdim,     // vector dimension
    GrB_Index nzmax,    // size of Ai and Ax for sparse/hypersparse
    GrB_Index nvals,    // # of entries for bitmap
    bool jumbled,       // if true, sparse/hypersparse may be jumbled
    GrB_Index nvec,     // size of Ah for hypersparse
    GrB_Index **Ap,     // pointers, size nvec+1 for hyper, vdim+1 for sparse
    GrB_Index **Ah,     // vector indices, size nvec for hyper
    int8_t **Ab,        // bitmap, size nzmax
    GrB_Index **Ai,     // indices, size nzmax
    void **Ax,          // values, size nzmax
    int sparsity,       // hypersparse, sparse, bitmap, or full
    bool is_csc,        // if true then matrix is by-column, else by-row
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    (*A) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    if (vlen > GxB_INDEX_MAX || nvals > GxB_INDEX_MAX ||
        vdim > GxB_INDEX_MAX || nzmax > GxB_INDEX_MAX)
    { 
        return (GrB_INVALID_VALUE) ;
    }

    if (sparsity == GxB_BITMAP || sparsity == GxB_FULL)
    {
        // ignore nzmax on input; compute it instead
        bool ok = GB_Index_multiply ((GrB_Index *) &nzmax, vlen, vdim) ;
        if (!ok)
        { 
            // problem too large
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    if (nzmax > 0)
    { 
        GB_RETURN_IF_NULL (Ax) ;
        GB_RETURN_IF_NULL (*Ax) ;
    }

    switch (sparsity)
    {
        case GxB_HYPERSPARSE : 
            GB_RETURN_IF_NULL (Ah) ;
            GB_RETURN_IF_NULL (*Ah) ;

        case GxB_SPARSE : 
            GB_RETURN_IF_NULL (Ap) ;
            GB_RETURN_IF_NULL (*Ap) ;
            if (nzmax > 0)
            {
                GB_RETURN_IF_NULL (Ai) ;
                GB_RETURN_IF_NULL (*Ai) ;
            }
            break ;
            
        case GxB_BITMAP : 
            if (nzmax > 0)
            {
                GB_RETURN_IF_NULL (Ab) ;
                GB_RETURN_IF_NULL (*Ab) ;
            }

        case GxB_FULL : 
        default: ;
    }

    //--------------------------------------------------------------------------
    // allocate just the header of the matrix, not the content
    //--------------------------------------------------------------------------

    GrB_Info info = GB_new (A, // any sparsity, new header
        type, vlen, vdim, GB_Ap_null, is_csc,
        sparsity, GB_Global_hyper_switch_get ( ), nvec, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        ASSERT ((*A) == NULL) ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    // transplant the user's content into the matrix
    (*A)->nzmax = nzmax ;
    (*A)->magic = GB_MAGIC ;

    switch (sparsity)
    {
        case GxB_HYPERSPARSE : 
            (*A)->nvec = nvec ;
            (*A)->h = (int64_t *) (*Ah) ; (*Ah) = NULL ;

        case GxB_SPARSE : 
            (*A)->jumbled = jumbled ;
            (*A)->nvec_nonempty = -1 ;
            (*A)->p = (int64_t *) (*Ap) ; (*Ap) = NULL ;
            (*A)->i = (int64_t *) (*Ai) ; (*Ai) = NULL ;
            break ;

        case GxB_BITMAP : 
            (*A)->nvals = nvals ;
            (*A)->b = (*Ab) ; (*Ab) = NULL ;

        case GxB_FULL : 
        default: ;
    }

    (*A)->x = (*Ax) ; (*Ax) = NULL ;

    //--------------------------------------------------------------------------
    // import is successful
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (*A, "A imported", GB0) ;
    return (GrB_SUCCESS) ;
}

