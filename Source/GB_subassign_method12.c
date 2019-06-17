//------------------------------------------------------------------------------
// GB_subassign_method12: C(I,J)<#M> += scalar ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method12
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
    const void *scalar,
    const GrB_Type atype,
    const GrB_Matrix S,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_MASK ;
    GB_GET_S ;
    GB_GET_ACCUM_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 12: C(I,J)<#M> += scalar ; using S
    //--------------------------------------------------------------------------

    // time:  O(nI*nJ)

    // PARALLEL: split nI*nJ into all-coarse or all-fine tasks.
    GBI3s_for_each_vector (S, M, scalar)
    {

        //----------------------------------------------------------------------
        // get S(:,j), M(:,j) and the scalar
        //----------------------------------------------------------------------

        GBI3s_jth_iteration (Iter, j, pS, pS_end, pM, pM_end) ;

        //----------------------------------------------------------------------
        // do a 3-way merge of S(:,j), M(:,j), and the scalar
        //----------------------------------------------------------------------

        // jC = J [j] ; or J is a colon expression
        int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

        // The merge is similar to GB_mask, except that it does not produce
        // another output matrix.  Instead, the results are written directly
        // into C, either modifying the entries there or adding pending tuples.

        // There are three sorted lists to merge:
        // S(:,j) in [pS .. pS_end-1]
        // M(:,j) in [pM .. pM_end-1]
        // A(:,j) an expanded scalar, an implicit dense vector.

        // The head of each list is at index pS, pA, and pM, and an entry is
        // 'discarded' by incrementing its respective index via GB_NEXT(.).
        // Once a list is consumed, a query for its next row index will result
        // in a dummy value nI larger than all valid row indices.

        //----------------------------------------------------------------------
        // while either list S(:,j) or A(:,j) have entries
        //----------------------------------------------------------------------

        // for each iA in I [...]:
        for (int64_t iA = 0 ; iA < nI ; iA++)
        {

            //------------------------------------------------------------------
            // Get the indices at the top of each list.
            //------------------------------------------------------------------

            // If a list has been consumed, use a dummy index nI
            // that is larger than all valid indices.
            int64_t iS = (pS < pS_end) ? Si [pS] : nI ;
            int64_t iM = (pM < pM_end) ? Mi [pM] : nI ;

            //------------------------------------------------------------------
            // find the smallest index of [iS iA iM]
            //------------------------------------------------------------------

            int64_t i = iA ;

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
            // handle all 6 cases
            //------------------------------------------------------------------

            if (i == iS)
            {
                ASSERT (i == iA) ;
                {
                    // both S (i,j) and A (i,j) present
                    GB_C_S_LOOKUP ;
                    if (mij)
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =C+A ): apply accum
                        // [X A 1]: action: ( undelete ): bring zombie back
                        GB_withaccum_C_A_1_scalar ;
                    }
                    else
                    { 
                        // ----[C A 0] or [X A 0]-------------------------------
                        // [X A 0]: action: ( X ): still a zombie
                        // [C A 0]: C_repl: action: ( delete ):zombie
                        // [C A 0]: no C_repl: action: ( C ): none
                        GB_C_A_0 ;
                    }
                    GB_NEXT (S) ;
                }
            }
            else
            {
                ASSERT (i == iA) ;
                {
                    // S (i,j) is not present, A (i,j) is present
                    if (mij)
                    { 
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
                        // iC = I [iA] ; or I is a colon expression
                        int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                        GB_D_A_1_scalar ;
                    }
                    else
                    { 
                        // ----[. A 0]------------------------------------------
                        // action: ( . ): no action
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

