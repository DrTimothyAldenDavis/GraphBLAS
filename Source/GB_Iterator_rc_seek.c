//------------------------------------------------------------------------------
// GB_Iterator_rc_seek: seek a row/col iterator to A(:,j) or to jth vector of A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Seek a row iterator to A(j,:), a col iterator to A(:,j).  If kth_vector is
// true, seek to the jth vector instead.  For sparse, bitmap, or full matrices,
// this is the same as A(j,:) for a row iterator or A(:,j) for a col iterator.
// It only affects how hypersparse matrices are traversed.

#include "GB.h"

GrB_Info GB_Iterator_rc_seek
(
    GxB_Iterator iterator,
    GrB_Index j,
    bool kth_vector
)
{

    //--------------------------------------------------------------------------
    // check if the iterator is exhausted
    //--------------------------------------------------------------------------

    if (j >= ((kth_vector) ? iterator->anvec : iterator->avdim))
    {
        iterator->k = iterator->anvec ;
        return (GxB_EXHAUSTED) ;
    }

    //--------------------------------------------------------------------------
    // attach the iterator to A(:,j)
    //--------------------------------------------------------------------------

    switch (iterator->A_sparsity)
    {
        default : 
        case GxB_SPARSE : 
        {
            // attach to A(:,j), which is also the jth vector of A
            iterator->pstart = iterator->Ap [j] ;
            iterator->pend = iterator->Ap [j+1] ;
            iterator->p = iterator->pstart ;
            iterator->k = j ;
        }
        break ;

        case GxB_HYPERSPARSE : 
        {
            if (kth_vector)
            {
                // attach to the jth vector of A; this is much faster than
                // searching Ah for the value j, to attach to A(:,j)
                iterator->pstart = iterator->Ap [j] ;
                iterator->pend = iterator->Ap [j+1] ;
                iterator->p = iterator->pstart ;
                iterator->k = j ;
            }
            else
            {
                // find the A(:,j) vector in Ah [0:anvec-1]
                int64_t k = 0 ;
                bool found ;
                const int64_t *restrict Ah = iterator->Ah ;
                if (j == 0)
                {
                    found = (Ah [0] == 0) ;
                }
                else
                {
                    int64_t pright = iterator->anvec-1 ;
                    GB_SPLIT_BINARY_SEARCH (j, Ah, k, pright, found) ;
                }
                if (found)
                {
                    // A(:,j) is the kth vector in the Ah hyperlist
                    iterator->pstart = iterator->Ap [k] ;
                    iterator->pend = iterator->Ap [k+1] ;
                    iterator->p = iterator->pstart ;
                    iterator->k = k ;
                }
                else
                {
                    // A(:,j) is not in the hyperlist; point the iterator to the
                    // vector that appears just before j, or -1 if j < Ah [0],
                    // so that seeking to the next vector with iterator->k++
                    // moves to the first vector larger than j.
                    iterator->pstart = 0 ;
                    iterator->pend = 0 ;
                    iterator->p = 0 ;
                    iterator->k = --k ;
                }
            }
        }
        break ;

        case GxB_BITMAP : 
        {
            // attach to A(:,j), which is also the jth vector of A
            iterator->pstart = j * iterator->avlen ;
            iterator->pend = (j+1) * iterator->avlen ;
            iterator->p = iterator->pstart ;
            iterator->k = j ;
            return (GB_Iterator_rc_bitmap_next (iterator)) ;
        }
        break ;

        case GxB_FULL : 
        {
            // attach to A(:,j), which is also the jth vector of A
            iterator->pstart = j * iterator->avlen ;
            iterator->pend = (j+1) * iterator->avlen ;
            iterator->p = iterator->pstart ;
            iterator->k = j ;
        }
        break ;
    }

    return ((iterator->p >= iterator->pend) ? GrB_NO_VALUE : GrB_SUCCESS) ;
}

