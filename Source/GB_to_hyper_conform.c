//------------------------------------------------------------------------------
// GB_to_hyper_conform: conform a matrix to its desired hypersparse format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input matrix can have shallow A->p and/or A->h components.  If the
// hypersparsity is changed, these components are no longer shallow.  If the
// method fails and the matrix is shallow, all content is removed or freed.
// Zombies are OK, but A never has pending tuples.  However, this function is
// agnostic about pending tuples so they could be OK.

#include "GB.h"

GrB_Info GB_to_hyper_conform    // conform a matrix to its desired format
(
    GrB_Matrix A,               // matrix to conform
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A to conform", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // convert to full if all entries present
    //--------------------------------------------------------------------------

    if (GB_IS_FULL (A))
    {
        // A is already full; nothing to do
        ASSERT (!GB_ZOMBIES (A)) ;
        ASSERT_MATRIX_OK (A, "A conformed: already full", GB0) ;
        return (GrB_SUCCESS) ;
    }

    if (GB_is_dense (A) && !GB_ZOMBIES (A))
    {
        // A is sparse or hypersparse with all entries present; convert to full
        GB_sparse_to_full (A) ;
        ASSERT_MATRIX_OK (A, "A conformed: converted to full", GB0) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // convert to sparse or hypersparse
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;

    if (A->nvec_nonempty < 0)
    { 
        A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    }

    if (A->h == NULL &&
        GB_to_hyper_test (A->hyper_ratio, A->nvec_nonempty, A->vdim))
    { 
        // A is sparse but should be converted to hypersparse
        info = GB_to_hyper (A, Context) ;
    }
    else if (A->h != NULL &&
        GB_to_nonhyper_test (A->hyper_ratio, A->nvec_nonempty, A->vdim))
    { 
        // A is hypersparse but should be converted to sparse
        ASSERT (!GB_IS_FULL (A)) ;
        info = GB_to_nonhyper (A, Context) ;
    }
    else
    { 
        // leave the matrix as-is
        ;
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory; all content has been freed
        ASSERT (A->magic == GB_MAGIC2) ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A conformed", GB0) ;
    return (GrB_SUCCESS) ;
}

