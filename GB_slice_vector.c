//------------------------------------------------------------------------------
// GB_slice_vector:  slice a vector for GB_add and GB_emult
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A(:,ka) and B(:,kb) are two long vectors that will be added with GB_add or
// GB_emult, and the work to compute them needs to be split into multiple
// tasks.  They represent the same vector index j, for C(:,j) = A(:,j) +
// B(:,j).  The vector index j is not needed here.  The vectors ka and kb are
// not required, either; just the positions where the vectors appear in A and B
// (pA_start, pA_end, pB_start, and pB_end).

// Find i so that nnz (A (i:end,ka)) + nnz (B (i:end,kb)) is roughly equal to
// target_work = target_fraction * (nnz (A (:,ka)) + nnz (B (:,kb))).  The
// entries in A(i:end,ka) start at position pA in Ai and Ax, and the entries in
// B(i:end,kb) start at position pB in Bi and Bx.

// If the resulting subtask is too small then it is returned as empty, and i =
// vlen.  If the subtask is too large, then it is returned with i = 0.  In both
// cases, the function returns false.

// If a subtask is found with roughly the correct amount of target work, then
// the task starts at A(i,ka) and B(i,kb), at position pA in Ai,Ax and position
// pB in Bi,Bx.

// If n = A->vlen = B->vlen, anz = nnz (A (:,ka)), and bnz = nnz (B (:,kb)),
// then the total time taken by this function is O(log(n)*(log(anz)+log(bnz)),
// or at most O((log(n)^2)).


#define GB_DEBUG

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
    const double target_fraction    // target work fraction (0 to 1, inclusive)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (target_fraction >= 0 && target_fraction <= 1) ;
    ASSERT (Ai != NULL && Bi != NULL) ;
    ASSERT (p_i != NULL && p_pA != NULL && p_pB != NULL) ;

    //--------------------------------------------------------------------------
    // get A (:,ka) and B (:,kb)
    //--------------------------------------------------------------------------

    int64_t anz = pA_end - pA_start ;
    int64_t bnz = pB_end - pB_start ;

    double total_work  = anz + bnz ;
    double target_work = target_fraction * total_work ;

    //--------------------------------------------------------------------------
    // special cases
    //--------------------------------------------------------------------------

    if (total_work - target_work < 1024)
    {
        // this subtask starts at the beginning of A(:,ka) and B(:,kb)
        (*p_i)  = 0 ;
        (*p_pA) = pA_start ;
        (*p_pB) = pB_start ;
    }
    else if (target_work < 1024)
    {
        // this subtask is empty
        (*p_i)  = vlen ;
        (*p_pA) = pA_end ;
        (*p_pB) = pB_end ;
    }

    //--------------------------------------------------------------------------
    // find i, pA, and pB for the start of this task
    //--------------------------------------------------------------------------

    // search for index i in the range ileft:iright, inclusive
    int64_t ileft  = 0 ;
    int64_t iright = vlen-1 ;

    while (true)
    {

        //----------------------------------------------------------------------
        // find the index i in the middle of ileft:iright
        //----------------------------------------------------------------------

        int64_t i = (ileft + iright) / 2 ;

        //----------------------------------------------------------------------
        // find where i appears in A(:,ka)
        //----------------------------------------------------------------------

        bool afound ;
        int64_t pA = pA_start ;
        int64_t apright = pA_end - 1 ;
        GB_BINARY_SPLIT_SEARCH (i, Ai, pA, apright, afound) ;
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

        bool bfound ;
        int64_t pB = pB_start ;
        int64_t bpright = pB_end - 1 ;
        GB_BINARY_SPLIT_SEARCH (i, Bi, pB, bpright, bfound) ;

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

        if (work < 0.8 * target_work)
        {

            //------------------------------------------------------------------
            // work is too low
            //------------------------------------------------------------------

            // work is too low, so i is too high.
            // Keep searching in the range (ileft:i), inclusive.

            iright = i ;

        }
        else if (work > 1.25 * target_work)
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
            (*p_i)  = i ;
            (*p_pA) = pA ;
            (*p_pB) = pB ;
            return ;
        }
    }
}

