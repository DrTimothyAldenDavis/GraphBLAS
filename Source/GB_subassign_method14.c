//------------------------------------------------------------------------------
// GB_subassign_method14: C(I,J)<#M> += A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method14
(
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    const GrB_Matrix S,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_MASK ;
    GB_GET_A ;
    GB_GET_S ;
    GB_GET_ACCUM ;

    //--------------------------------------------------------------------------
    // Method 14: C(I,J)<#M> += A ; using S
    //--------------------------------------------------------------------------

    // GB_accum_mask case: C(:,:)<M> = A

    // PARALLEL: split S,M,A into coarse/fine tasks.  Same as Method 13.

    GBI3_for_each_vector (S, M, A)
    {

        //----------------------------------------------------------------------
        // do a 3-way merge of S(:,j), M(:,j), and A(:,j)
        //----------------------------------------------------------------------

        GBI3_jth_iteration (Iter, j,
            pS, pS_end, pM, pM_end, pA, pA_end) ;
        // jC = J [j] ; or J is a colon expression
        int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

        // The merge is similar to GB_mask, except that it does not produce
        // another output matrix.  Instead, the results are written directly
        // into C, either modifying the entries there or adding pending tuples.

        // There are three sorted lists to merge:
        // S(:,j) in [pS .. pS_end-1]
        // M(:,j) in [pM .. pM_end-1]
        // A(:,j) in [pA .. pA_end-1]

        // The head of each list is at index pS, pA, and pM, and an entry is
        // 'discarded' by incrementing its respective index via GB_NEXT(.).
        // Once a list is consumed, a query for its next row index will result
        // in a dummy value nI larger than all valid row indices.

        //----------------------------------------------------------------------
        // while either list S(:,j) or A(:,j) have entries
        //----------------------------------------------------------------------

        while (pS < pS_end || pA < pA_end)
        {

            //------------------------------------------------------------------
            // Get the indices at the top of each list.
            //------------------------------------------------------------------

            // If a list has been consumed, use a dummy index nI
            // that is larger than all valid indices.
            int64_t iS = (pS < pS_end) ? Si [pS] : nI ;
            int64_t iA = (pA < pA_end) ? Ai [pA] : nI ;
            int64_t iM = (pM < pM_end) ? Mi [pM] : nI ;

            //------------------------------------------------------------------
            // find the smallest index of [iS iA iM]
            //------------------------------------------------------------------

            // i = min ([iS iA iM])
            int64_t i = GB_IMIN (iS, GB_IMIN (iA, iM)) ;
            ASSERT (i < nI) ;

            //------------------------------------------------------------------
            // get M(i,j)
            //------------------------------------------------------------------

            // If an explicit value of M(i,j) must be tested, it must first be
            // typecasted to bool.  If (i == iM), then M(i,j) is present and is
            // typecasted into mij and then discarded.  Otherwise, if M(i,j) is
            // not present, mij is set to false.

            bool mij ;
            if (i == iM)
            { 
                // mij = (bool) M [pM]
                cast_M (&mij, Mx +(pM*msize), 0) ;
                GB_NEXT (M) ;
            }
            else
            { 
                // mij not present, implicitly false
                ASSERT (i < iM) ;
                mij = false ;
            }

            // explicitly complement the mask entry mij
            if (Mask_comp)
            { 
                mij = !mij ;
            }

            //------------------------------------------------------------------
            // handle all 12 cases
            //------------------------------------------------------------------

            if (i == iS)
            {
                if (i == iA)
                {
                    // both S (i,j) and A (i,j) present
                    GB_C_S_LOOKUP ;
                    if (mij)
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =A ): A to C no accum
                        // [C A 1]: action: ( =C+A ): apply accum
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_withaccum_C_A_1_matrix ;
                    }
                    else
                    { 
                        // ----[C A 0] or [X A 0]-------------------------------
                        // [X A 0]: action: ( X ): still a zombie
                        // [C A 0]: C_repl: action: ( delete ): zombie
                        // [C A 0]: no C_repl: action: ( C ): none
                        GB_C_A_0 ;
                    }
                    GB_NEXT (S) ;
                    GB_NEXT (A) ;
                }
                else
                {
                    // S (i,j) is present but A (i,j) is not
                    GB_C_S_LOOKUP ;
                    if (mij)
                    { 
                        // ----[C . 1] or [X . 1]-------------------------------
                        // [C . 1]: action: ( C ): no change w accum
                        // [X . 1]: action: ( X ): still a zombie
                        // withaccum_C_D_1_matrix ;
                    }
                    else
                    { 
                        // ----[C . 0] or [X . 0]-------------------------------
                        // [X . 0]: action: ( X ): still a zombie
                        // [C . 0]: if C_repl: action: ( delete ): zombie
                        // [C . 0]: no C_repl: action: ( C ): none
                        GB_C_D_0 ;
                    }
                    GB_NEXT (S) ;
                }
            }
            else
            {
                if (i == iA)
                {
                    // S (i,j) is not present, A (i,j) is present
                    if (mij)
                    { 
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
                        // iC = I [iA] ; or I is a colon expression
                        int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                        GB_D_A_1_matrix ;
                    }
                    else
                    { 
                        // ----[. A 0]------------------------------------------
                        // action: ( . ): no action
                    }
                    GB_NEXT (A) ;
                }
                else
                { 
                    // neither S (i,j) nor A (i,j) present
                    // ----[. . 1]----------------------------------------------
                    // ----[. . 0]----------------------------------------------
                    // action: ( . ): no action
                    ASSERT (i == iM) ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

