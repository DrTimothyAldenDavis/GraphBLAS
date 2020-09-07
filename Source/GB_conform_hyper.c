//------------------------------------------------------------------------------
// GB_conform_hyper: conform a sparse matrix to its desired hypersparse format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input matrix must be sparse or hypersparse, and it may be left as-is,
// or converted to sparse/hypersparse.

// The input matrix can have shallow A->p and/or A->h components.  If the
// hypersparsity is changed, these components are no longer shallow.  If the
// method fails and the matrix is shallow, all content is removed or freed.

#include "GB.h"

#define GB_FREE_ALL GB_phbix_free (A) ;

GrB_Info GB_conform_hyper       // conform a matrix to sparse/hypersparse
(
    GrB_Matrix A,               // matrix to conform
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;
    ASSERT_MATRIX_OK (A, "A to conform_hyper", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;      // ok: method not called for A full
    ASSERT (!GB_IS_BITMAP (A)) ;    // ok: method not called for A bitmap
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // convert to sparse or hypersparse
    //--------------------------------------------------------------------------

    if (A->nvec_nonempty < 0)
    { 
        A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    }

    if (A->h == NULL && GB_convert_sparse_to_hyper_test (A->hyper_switch,
        A->nvec_nonempty, A->vdim))
    { 
        // A is sparse but should be converted to hypersparse
        info = GB_convert_sparse_to_hyper (A, Context) ;
    }
    else if (A->h != NULL && GB_convert_hyper_to_sparse_test (A->hyper_switch,
        A->nvec_nonempty, A->vdim))
    { 
        // A is hypersparse but should be converted to sparse
        info = GB_convert_hyper_to_sparse (A, Context) ;
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

    ASSERT_MATRIX_OK (A, "A conform_hyper result", GB0) ;
    return (GrB_SUCCESS) ;
}

