//------------------------------------------------------------------------------
// GB_slice_vector:  slice a vector for GB_add, GB_emult, and GB_mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A(:,kA) and B(:,kB) are two long vectors that will be added with GB_add,
// GB_emult, or GB_mask, and the work to compute them needs to be split into
// multiple tasks.  They represent the same vector index j, for:

//      C(:,j) = A(:,j) +  B(:,j) in GB_add
//      C(:,j) = A(:,j) .* B(:,j) in GB_emult
//      C(:,j)<M(:,j)> = B(:,j) in GB_mask (A is passed in as the input C)

// The vector index j is not needed here.  The vectors kA and kB are not
// required, either; just the positions where the vectors appear in A and B
// (pA_start, pA_end, pB_start, and pB_end).

// This method finds i so that nnz (A (i:end,kA)) + nnz (B (i:end,kB)) is
// roughly equal to target_work.  The entries in A(i:end,kA) start at position
// pA in Ai and Ax, and the entries in B(i:end,kB) start at position pB in Bi
// and Bx.  Once the work is split, pM is found for M(i:end,kM), if the mask M
// is present.

// If n = A->vlen = B->vlen, anz = nnz (A (:,kA)), and bnz = nnz (B (:,kB)),
// then the total time taken by this function is O(log(n)*(log(anz)+log(bnz))),
// or at most O((log(n)^2)).

#include "GB.h"

void GB_slice_vector
(
    // output: return i, pA, and pB
    int64_t *p_i,                   // work starts at A(i,kA) and B(i,kB)
    int64_t *p_pM,                  // M(i:end,kM) starts at pM
    int64_t *p_pA,                  // A(i:end,kA) starts at pA
    int64_t *p_pB,                  // B(i:end,kB) starts at pB
    // input:
    const int64_t pM_start,         // M(:,kM) starts at pM_start in Mi,Mx
    const int64_t pM_end,           // M(:,kM) ends at pM_end-1 in Mi,Mx
    const int64_t *restrict Mi,     // indices of M (or NULL)
    const int64_t pA_start,         // A(:,kA) starts at pA_start in Ai,Ax
    const int64_t pA_end,           // A(:,kA) ends at pA_end-1 in Ai,Ax
    const int64_t *restrict Ai,     // indices of A
    const int64_t pB_start,         // B(:,kB) starts at pB_start in Bi,Bx
    const int64_t pB_end,           // B(:,kB) ends at pB_end-1 in Bi,Bx
    const int64_t *restrict Bi,     // indices of B
    const int64_t vlen,             // A->vlen and B->vlen
    const double target_work        // target work
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Ai != NULL && Bi != NULL) ;
    ASSERT (p_i != NULL && p_pM != NULL && p_pA != NULL && p_pB != NULL) ;

    //--------------------------------------------------------------------------
    // find i, pA, and pB for the start of this task
    //--------------------------------------------------------------------------

    // TODO allow ileft and iright to be specified on input, to limit the
    // search.

    // search for index i in the range ileft:iright, inclusive
    int64_t ileft  = 0 ;
    int64_t iright = vlen-1 ;
    int64_t i = 0 ;
    int64_t pM = pM_start ;
    int64_t pA = pA_start ;
    int64_t pB = pB_start ;
    int64_t aknz = pA_end - pA_start ;
    int64_t bknz = pB_end - pB_start ;
    int64_t mknz = pM_end - pM_start ;      // zero if M not present

    while (ileft < iright)
    {

        //----------------------------------------------------------------------
        // find the index i in the middle of ileft:iright
        //----------------------------------------------------------------------

        i = (ileft + iright) / 2 ;
        // printf ("   slice vector i "GBd" in ["GBd" to "GBd"]\n", i, ileft,
        //     iright) ;

        //----------------------------------------------------------------------
        // find where i appears in A(:,kA)
        //----------------------------------------------------------------------

        double awork = 0 ;
        pA = pA_start ;
        if (aknz == vlen)
        { 
            // A(:,kA) is dense; no need for a binary search
            pA = pA_start + i ;
            ASSERT (Ai [pA] == i) ;
        }
        else if (aknz > 0)
        { 
            bool afound ;
            int64_t apright = pA_end - 1 ;
            GB_BINARY_SPLIT_SEARCH (i, Ai, pA, apright, afound) ;
        }
        if (pA >  0) ASSERT (Ai [pA-1] < i) ;
        if (pA >= 0 && pA < pA_end) ASSERT (Ai [pA] >= i) ;

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
        // find where i appears in B(:,kB)
        //----------------------------------------------------------------------

        double bwork = 0 ;
        pB = pB_start ;
        if (bknz == vlen)
        { 
            // B(:,kB) is dense; no need for a binary search
            pB = pB_start + i ;
            ASSERT (Bi [pB] == i) ;
        }
        else if (bknz > 0)
        { 
            bool bfound ;
            int64_t bpright = pB_end - 1 ;
            GB_BINARY_SPLIT_SEARCH (i, Bi, pB, bpright, bfound) ;
        }
        if (pB >  0) ASSERT (Bi [pB-1] < i) ;
        if (pB >= 0 && pB < pB_end) ASSERT (Bi [pB] >= i) ;

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
    // find where i appears in M(:,kM)
    //--------------------------------------------------------------------------

    if (Mi != NULL)
    {
        pM = pM_start ;
        if (mknz == vlen)
        { 
            // M(:,kM) is dense; no need for a binary search
            pM = pM_start + i ;
            ASSERT (Mi [pM] == i) ;
        }
        else if (mknz > 0)
        { 
            bool mfound ;
            int64_t mpright = pM_end - 1 ;
            GB_BINARY_SPLIT_SEARCH (i, Mi, pM, mpright, mfound) ;
        }
        if (pM >  0) ASSERT (Mi [pM-1] < i) ;
        if (pM >= 0 && pM < pM_end) ASSERT (Mi [pM] >= i) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_i)  = i ;
    (*p_pM) = pM ;
    (*p_pA) = pA ;
    (*p_pB) = pB ;
}

