//------------------------------------------------------------------------------
// GB_sparse_to_full: convert a matrix from sparse/hypersparse to full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GB_PUBLIC                       // used by MATLAB interface
void GB_sparse_to_full          // convert matrix from sparse to full
(
    GrB_Matrix A                // matrix to convert from sparse to full
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting sparse to full", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (GB_is_dense (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    GBBURBLE ("(sparse to full) ") ;

    //--------------------------------------------------------------------------
    // free A->h, A->p, and A->i
    //--------------------------------------------------------------------------

    GB_ph_free (A) ;

    if (!A->i_shallow) GB_FREE (A->i) ;
    A->i = NULL ;
    A->i_shallow = false ;

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;

    A->plen = -1 ;
    A->nvec = avdim ;
    A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from sparse to full", GB0) ;
    ASSERT (GB_IS_FULL (A)) ;
}

