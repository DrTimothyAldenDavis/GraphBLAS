//------------------------------------------------------------------------------
// GB_convert_any_to_full: convert any matrix to full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// All entries must be present; GB_is_dense (A) must be true on input,
// and the matrix cannot have any pending work.

// A may be hypersparse, sparse, bitmap, or full on input.
// A is full on output.

// OK: BITMAP

#include "GB.h"

GB_PUBLIC                       // used by MATLAB interface
void GB_convert_any_to_full     // convert any matrix to full
(
    GrB_Matrix A                // matrix to convert to full
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting any to full", GB0) ;
    ASSERT (GB_is_dense (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    if (GB_IS_FULL (A))
    { 
        // already full; nothing to do
        return ;
    }

    ASSERT (!GB_IS_FULL (A)) ;

    GBURBLE ("(%s to full) ", (A->h != NULL) ? "hypersparse" :
        (GB_IS_BITMAP (A) ? "bitmap" : "sparse")) ;

    //--------------------------------------------------------------------------
    // free A->h, A->p, A->i, and A->b
    //--------------------------------------------------------------------------

    GB_ph_free (A) ;

    if (!A->i_shallow) GB_FREE (A->i) ;
    A->i = NULL ;
    A->i_shallow = false ;

    if (!A->b_shallow) GB_FREE (A->b) ;
    A->b = NULL ;
    A->b_shallow = false ;

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;

    A->plen = -1 ;
    A->nvec = avdim ;
    A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from any to full", GB0) ;
    ASSERT (GB_IS_FULL (A)) ;
}

