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

    // the 5 arrays:
    GrB_Index **Ap,     // pointers, for sparse and hypersparse formats.
                        // Ap_size >= nvec+1 for hyper, Ap_size >= vdim+1 for
                        // sparse.  Ignored for bitmap and full formats.
    GrB_Index Ap_size,  // size of Ap; ignored if Ap is ignored.
    GrB_Index **Ah,     // vector indices, Ah_size >= nvec for hyper.
                        // Ignored for sparse, bitmap, and full formats.
    GrB_Index Ah_size,  // size of Ah; ignored if Ah is ignored.
    int8_t **Ab,        // bitmap, for bitmap format only, Ab_size >= vlen*vdim.
                        // Ignored for hyper, sparse, and full formats.  
    GrB_Index Ab_size,  // size of Ab; ignored if Ab is ignored.
    GrB_Index **Ai,     // indices, size Ai_size >= nvals(A) for hyper and
                        // sparse formats.  Ignored for bitmap and full.
    GrB_Index Ai_size,  // size of Ai; ignored if Ai is ignored.
    void **Ax,          // values, Ax_size is either 1, or >= nvals(A) for
                        // hyper or sparse formats.  Ax_size >= vlen*vdim for
                        // bitmap or full formats.
    GrB_Index Ax_size,  // size of Ax; never ignored.

    // additional information for specific formats:
    GrB_Index nvals,    // # of entries for bitmap format.
    bool jumbled,       // if true, sparse/hypersparse may be jumbled.
    GrB_Index nvec,     // size of Ah for hypersparse format.

    // information for all formats:
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
    if (vlen  > GxB_INDEX_MAX || vdim > GxB_INDEX_MAX ||
        nvals > GxB_INDEX_MAX || nvec > GxB_INDEX_MAX ||
        Ap_size > GxB_INDEX_MAX ||
        Ah_size > GxB_INDEX_MAX || Ab_size > GxB_INDEX_MAX ||
        Ai_size > GxB_INDEX_MAX || Ax_size > GxB_INDEX_MAX)
    { 
        return (GrB_INVALID_VALUE) ;
    }

    // full_size = vlen*vdim, for bitmap and full formats
    bool ok = true ;
    int64_t full_size ;
    if (sparsity == GxB_BITMAP || sparsity == GxB_FULL)
    {
        ok = GB_Index_multiply ((GrB_Index *) &full_size, vlen, vdim) ;
        if (!ok)
        { 
            // problem too large: only Ax_size == 1 is possible
            full_size = 1 ;
        }
    }

    if (Ax_size > 0)
    { 
        GB_RETURN_IF_NULL (Ax) ;
        GB_RETURN_IF_NULL (*Ax) ;
    }

    switch (sparsity)
    {
        case GxB_HYPERSPARSE : 
            GB_RETURN_IF_NULL (Ah) ;
            GB_RETURN_IF_NULL (*Ah) ;
            if (Ap_size < nvec+1) return (GrB_INVALID_VALUE) ;
            if (Ah_size < nvec)   return (GrB_INVALID_VALUE) ;
            if (nvec > vdim)      return (GrB_INVALID_VALUE) ;
            GB_RETURN_IF_NULL (Ap) ;
            GB_RETURN_IF_NULL (*Ap) ;
            if (Ai_size > 0)
            {
                GB_RETURN_IF_NULL (Ai) ;
                GB_RETURN_IF_NULL (*Ai) ;
            }
            nvals = (*Ap) [nvec] ;
            // printf ("Ai_size %ld nvals %ld\n", Ai_size, nvals) ;
            if (Ai_size < nvals)  return (GrB_INVALID_VALUE) ;
            if (Ax_size < nvals)  return (GrB_INVALID_VALUE) ;
            break ;

        case GxB_SPARSE : 

            GB_RETURN_IF_NULL (Ap) ;
            GB_RETURN_IF_NULL (*Ap) ;
            if (Ai_size > 0)
            {
                GB_RETURN_IF_NULL (Ai) ;
                GB_RETURN_IF_NULL (*Ai) ;
            }
            // printf ("Ap_size %ld vdim %ld\n", Ap_size, vdim) ;
            if (Ap_size < vdim+1) return (GrB_INVALID_VALUE) ;
            nvals = (*Ap) [vdim] ;
            if (Ai_size < nvals)  return (GrB_INVALID_VALUE) ;
            if (Ax_size < nvals)  return (GrB_INVALID_VALUE) ;
            break ;

        case GxB_BITMAP : 
            if (Ab_size > 0)
            {
                GB_RETURN_IF_NULL (Ab) ;
                GB_RETURN_IF_NULL (*Ab) ;
            }
            if (!ok) return (GrB_OUT_OF_MEMORY) ;
            if (nvals > full_size)   return (GrB_INVALID_VALUE) ;
            if (Ab_size < full_size) return (GrB_INVALID_VALUE) ;

        case GxB_FULL : 
            if (Ax_size > 1 && Ax_size < full_size) return (GrB_INVALID_VALUE) ;
            break ;

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
    (*A)->magic = GB_MAGIC ;

    // TODO: keep Ap_size, Ah_size, Ab_size, Ai_size, Ax_size in the
    // GrB_Matrix data structure, and remove A->nzmax.

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
            (*A)->nzmax = GB_IMIN (Ai_size, Ax_size) ;
            break ;

        case GxB_BITMAP : 
            (*A)->nvals = nvals ;
            (*A)->b = (*Ab) ; (*Ab) = NULL ;
            (*A)->nzmax = GB_IMIN (Ab_size, Ax_size) ;
            break ;

        case GxB_FULL : 
            (*A)->nzmax = Ax_size ;
            break ;

        default: ;
    }

    (*A)->x = (*Ax) ; (*Ax) = NULL ;

    //--------------------------------------------------------------------------
    // import is successful
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (*A, "A imported", GB0) ;
    return (GrB_SUCCESS) ;
}

