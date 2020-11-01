//------------------------------------------------------------------------------
// GB_convert_to_full: convert a matrix to full; deleting prior values
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_convert_to_full     // convert matrix to full; delete prior values
(
    GrB_Matrix A                // matrix to convert to full
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting to full", GB0) ;
    GBURBLE ("(to full) ") ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;
    ASSERT (GB_IS_FULL (A) || GB_IS_BITMAP (A) || GB_IS_SPARSE (A) ||
        GB_IS_HYPERSPARSE (A)) ;

    //--------------------------------------------------------------------------
    // free all prior content
    //--------------------------------------------------------------------------

    GB_phbix_free (A) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;

    GrB_Index anzmax ;
    bool ok = GB_Index_multiply (&anzmax, avlen, avdim) ;
    if (!ok)
    {   GB_cov[3088]++ ;
// NOT COVERED (3088):
GB_GOTCHA ;
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // allocate new space for A->x
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    // in debug mode, calloc the matrix so it can be safely printed below
    A->x = GB_CALLOC (anzmax * A->type->size, GB_void) ;    // BIG (debug)
    #else
    // in production mode, A->x is uninitialized
    A->x = GB_MALLOC (anzmax * A->type->size, GB_void) ;
    #endif

    if (A->x == NULL)
    {   GB_cov[3089]++ ;
// covered (3089): 524
        // out of memory
        GB_phbix_free (A) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    A->plen = -1 ;
    A->nvec = avdim ;
    A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;   // TODO::OK

    A->nzmax = GB_IMAX (anzmax, 1) ;

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted to full (values all zero)", GB0) ;
    ASSERT (GB_IS_FULL (A)) ;
    return (GrB_SUCCESS) ;
}

