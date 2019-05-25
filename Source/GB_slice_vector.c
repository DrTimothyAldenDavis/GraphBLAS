//------------------------------------------------------------------------------
// GB_slice_vector:  slice a vector for GB_add, GB_emult, and GB_mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A(:,ka) and B(:,kb) are two long vectors that will be added with GB_add,
// GB_emult, or GB_mask, and the work to compute them needs to be split into
// multiple tasks.  They represent the same vector index j, for:

//      C(:,j) = A(:,j) +  B(:,j) in GB_add
//      C(:,j) = A(:,j) .* B(:,j) in GB_emult
//      C(:,j)<M(:,j)> = B(:,j) in GB_mask (A is passed in as the input C)

// The vector index j is not needed here.  The vectors ka and kb are not
// required, either; just the positions where the vectors appear in A and B
// (pA_start, pA_end, pB_start, and pB_end).

// This method finds i so that nnz (A (i:end,ka)) + nnz (B (i:end,kb)) is
// roughly equal to target_work.  The entries in A(i:end,ka) start at position
// pA in Ai and Ax, and the entries in B(i:end,kb) start at position pB in Bi
// and Bx.

// If n = A->vlen = B->vlen, anz = nnz (A (:,ka)), and bnz = nnz (B (:,kb)),
// then the total time taken by this function is O(log(n)*(log(anz)+log(bnz))),
// or at most O((log(n)^2)).

#include "GB.h"

void GB_slice_vector
(
    // output: return i, pA, and pB
    int64_t *p_i,                   // work starts at A(i,ka) and B(i,kb)
    int64_t *p_pA,                  // A(i:end,ka) starts at pA
    int64_t *p_pB,                  // B(i:end,kb) starts at pB
    // input:
    const int64_t pA_start,         // A(:,ka) starts at pA_start in Ai,Ax
    const int64_t pA_end,           // A(:,ka) ends at pA_end-1 in Ai,Ax
    const int64_t *restrict Ai,     // indices of A
    const int64_t pB_start,         // B(:,kb) starts at pB_start in Bi,Bx
    const int64_t pB_end,           // B(:,kb) ends at pB_end-1 in Bi,Bx
    const int64_t *restrict Bi,     // indices of B
    const int64_t vlen,             // A->vlen and B->vlen
    const double target_work        // target work
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Ai != NULL && Bi != NULL) ;
    ASSERT (p_i != NULL && p_pA != NULL && p_pB != NULL) ;

    //--------------------------------------------------------------------------
    // find i, pA, and pB for the start of this task
    //--------------------------------------------------------------------------

    // TODO allow ileft and iright to be specified on input, to limit the
    // search.

    // search for index i in the range ileft:iright, inclusive
    int64_t ileft  = 0 ;
    int64_t iright = vlen-1 ;
    int64_t i = 0 ;
    int64_t pA = pA_start ;
    int64_t pB = pB_start ;
    int64_t aknz = pA_end - pA_start ;
    int64_t bknz = pB_end - pB_start ;

    while (ileft < iright)
    {

        //----------------------------------------------------------------------
        // find the index i in the middle of ileft:iright
        //----------------------------------------------------------------------

        i = (ileft + iright) / 2 ;
        // printf ("   slice vector i "GBd" in ["GBd" to "GBd"]\n", i, ileft,
        //     iright) ;

        //----------------------------------------------------------------------
        // find where i appears in A(:,ka)
        //----------------------------------------------------------------------

        double awork = 0 ;
        pA = pA_start ;
        if (aknz == vlen)
        { 
            // A(:,ka) is dense; no need for a binary search
            pA = pA_start + i ;
            ASSERT (Ai [pA] == i) ;
        }
        else if (aknz > 0)
        { 
            bool afound ;
            int64_t apright = pA_end - 1 ;
            GB_BINARY_SPLIT_SEARCH (i, Ai, pA, apright, afound) ;
        }
        if (pA > 0) ASSERT (Ai [pA-1] < i) ;
        ASSERT (Ai [pA] >= i) ;

        // Ai has been split.  If afound is false:
        //      Ai [pA_start : pA-1] < i
        //      Ai [pA : pA_end-1]   > i
        // If afound is true:
        //      Ai [pA_start : pA-1] < i
        //      Ai [pA : pA_end-1]  >= i
        //
        // in both cases, if i is chosen as the breakpoint, then the
        // subtask starts at index i, and position pA in Ai,Ax.

        //----------------------------------------------------------------------
        // find where i appears in B(:,kb)
        //----------------------------------------------------------------------

        double bwork = 0 ;
        pB = pB_start ;
        if (bknz == vlen)
        { 
            // B(:,kb) is dense; no need for a binary search
            pB = pB_start + i ;
            ASSERT (Bi [pB] == i) ;
        }
        else if (bknz > 0)
        { 
            bool bfound ;
            int64_t bpright = pB_end - 1 ;
            GB_BINARY_SPLIT_SEARCH (i, Bi, pB, bpright, bfound) ;
        }
        if (pB > 0) ASSERT (Bi [pB-1] < i) ;
        ASSERT (Bi [pB] >= i) ;

        // Bi has been split.  If bfound is false:
        //      Bi [pB_start : pB-1] < i
        //      Bi [pB : pB_end-1]   > i
        // If bfound is true:
        //      Bi [pB_start : pB-1] < i
        //      Bi [pB : pB_end-1]  >= i
        //
        // in both cases, if i is chosen as the breakpoint, then the
        // subtask starts at index i, and position pB in Bi,Bx.

        //----------------------------------------------------------------------
        // determine if the subtask is near the target task size
        //----------------------------------------------------------------------

        double work = (pA_end - pA) + (pB_end - pB) ;
        // printf ("    work %g target %g\n", work, target_work) ;

        if (work < 0.9999 * target_work)
        { 

            //------------------------------------------------------------------
            // work is too low
            //------------------------------------------------------------------

            // work is too low, so i is too high.
            // Keep searching in the range (ileft:i), inclusive.

            iright = i ;

        }
        else if (work > 1.0001 * target_work)
        { 

            //------------------------------------------------------------------
            // work is too high
            //------------------------------------------------------------------

            // work is too high, so i is too low.
            // Keep searching in the range (i+1):iright, inclusive.

            ileft = i + 1 ;

        }
        else
        { 

            //------------------------------------------------------------------
            // work is about right; use this result.
            //------------------------------------------------------------------

            // return i, pA, and pB as the start of this task.
            ASSERT (0 <= i && i <= vlen) ;
            ASSERT (pA_start <= pA && pA <= pA_end) ;
            ASSERT (pB_start <= pB && pB <= pB_end) ;
            break ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_i)  = i ;
    (*p_pA) = pA ;
    (*p_pB) = pB ;
}

