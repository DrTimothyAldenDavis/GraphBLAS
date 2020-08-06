//------------------------------------------------------------------------------
// GB_conform: conform any matrix to its desired sparsity structure
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

#define GB_FREE_ALL GB_phbix_free (A) ;

GrB_Info GB_conform             // conform a matrix to its desired structure
(
    GrB_Matrix A,               // matrix to conform
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;
    ASSERT_MATRIX_OK (A, "A to conform", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    // ensure that vectors are never converted to hypersparse
    if (A->sparsity == GxB_HYPERSPARSE && A->vdim <= 1)
    { 
        A->sparsity = GxB_SPARSE ;
    }

    //--------------------------------------------------------------------------
    // select the sparsity structure
    //--------------------------------------------------------------------------

    switch (A->sparsity)
    { 

        //----------------------------------------------------------------------
        // always hypersparse
        //----------------------------------------------------------------------

        case GxB_HYPERSPARSE :

            A->hyper_switch = GB_ALWAYS_HYPER ;
            GB_OK (GB_convert_any_to_hyper (A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // always sparse
        //----------------------------------------------------------------------

        case GxB_SPARSE :

            A->hyper_switch = GB_NEVER_HYPER ;
            GB_OK (GB_convert_any_to_sparse (A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // always bitmap
        //----------------------------------------------------------------------

        case GxB_BITMAP :

            // sparse and hypersparse matrices may have zombies, pending tuples,
            // and may be jumbled.  Pending tuples must be assembled, but
            // zombies and jumbled are OK.
            A->hyper_switch = GB_NEVER_HYPER ;
            GB_MATRIX_WAIT_IF_PENDING (A) ;
            GB_OK (GB_convert_any_to_bitmap (A, Context)) ;
            break ;

        //---------------------------------------------------------------------
        // always full (or bitmap if not all entries present)
        //---------------------------------------------------------------------

        case GxB_FULL :

            // sparse and hypersparse matrices may have zombies, pending tuples,
            // and may be jumbled.  All pending work must be finished now.
            A->hyper_switch = GB_NEVER_HYPER ;
            GB_MATRIX_WAIT (A) ;
            if (GB_is_dense (A))
            {
                // all entries are present: convert to full
                GB_convert_any_to_full (A) ;
            }
            else
            {
                // some entries not present: convert to bitmap
                // TODO: use bitmap instead
                GB_OK (GB_convert_any_to_sparse (A, Context)) ;
                // GB_OK (GB_convert_any_to_bitmap (A, Context)) ;
            }
            break ;

        //----------------------------------------------------------------------
        // default: automatic selection of sparsity
        //----------------------------------------------------------------------

        default :
        case GxB_DEFAULT :

            if (GB_IS_FULL (A))
            {
                // A is already full; nothing to do
                ASSERT_MATRIX_OK (A, "A conformed: already full", GB0) ;
                ASSERT (!GB_ZOMBIES (A)) ;
                ASSERT (!GB_JUMBLED (A)) ;
                ASSERT (!GB_PENDING (A)) ;
                return (GrB_SUCCESS) ;
            }

            // A is now hypersparse, sparse, or bitmap; check for conversion to
            // full

            if (GB_is_dense (A) && !GB_ZOMBIES (A) && !(A->jumbled)
                && !GB_PENDING (A))
            { 
                // A is sparse, hypersparse, or bitmmap with all entries
                // present; convert to full.  A bitmap matrix cannot have
                // zombies, pending tuples, or be jumbled, so this step
                // converts any dense bitmap matrix to full.  The conversion
                // cannot be done if A has any pending work.
                ASSERT_MATRIX_OK (A, "A conformed: converting full", GB0) ;
                GB_convert_any_to_full (A) ;
                ASSERT_MATRIX_OK (A, "A conformed: converted to full", GB0) ;
                return (GrB_SUCCESS) ;
            }

            //------------------------------------------------------------------
            // convert to/from bitmap
            //------------------------------------------------------------------

            // A is now hypersparse, sparse, or bitmap, with some entries not
            // present or with pending work (or both).

            if (GB_IS_BITMAP (A))
            { 
                // A is a bitmap matrix with some entries not present.  TODO:
                // for now, leave as bitmap.  Instead, if nnz(A) is very small,
                // convert to sparse
                ASSERT_MATRIX_OK (A, "A conformed: already bitmap", GB0) ;
                return (GrB_SUCCESS) ;
            }

            // TODO: if nnz(A) is large compared to avlen*avdim, convert to
            // bitmap and return result

            //------------------------------------------------------------------
            // convert between sparse/hypersparse
            //------------------------------------------------------------------

            // A is now sparse or hypersparse; convert between the two as
            // desired.  All pending work is left pending.

            ASSERT (!GB_IS_FULL (A)) ;
            ASSERT (!GB_IS_BITMAP (A)) ;
            ASSERT (GB_ZOMBIES_OK (A)) ;
            ASSERT (GB_JUMBLED_OK (A)) ;
            ASSERT (GB_PENDING_OK (A)) ;
            GB_OK (GB_conform_hyper (A, Context)) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A conformed", GB0) ;
    return (GrB_SUCCESS) ;
}

