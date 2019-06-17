//------------------------------------------------------------------------------
// GB_subassign_method4: C(I,J)<!M> += scalar ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method4
(
    GrB_Matrix C,
    // input:
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type atype,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_MASK ;
    GB_GET_ACCUM_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 4: C(I,J)<!M> += scalar ; no S
    //--------------------------------------------------------------------------

    // The case for C(I,J)<M> += scalar, not using S, and C_replace
    // false already handled by the C_Mask_scalar case.

    // PARALLEL: split nI*nJ into all-coarse or all-fine
    GBI2s_for_each_vector (M, scalar)
    {

        //----------------------------------------------------------------------
        // get M(:,j) and the scalar
        //----------------------------------------------------------------------

        GBI2s_jth_iteration (Iter, j, pM, pM_end) ;

        //----------------------------------------------------------------------
        // get the C(:,jC) vector where jC = J [j]
        //----------------------------------------------------------------------

        int64_t GB_jC_LOOKUP ;

        if (pC_end - pC_start == cvlen)
        {

            //------------------------------------------------------------------
            // C(:,jC) is dense so binary search of C is not needed
            //------------------------------------------------------------------

            // for each iA in I [...]:
            for (int64_t iA = 0 ; iA < nI ; iA++)
            {

                //--------------------------------------------------------------
                // find M(iA,j)
                //--------------------------------------------------------------

                bool mij ;
                bool found = (pM < pM_end) && (Mi [pM] == iA) ;
                if (found)
                { 
                    // found it
                    cast_M (&mij, Mx +(pM*msize), 0) ;
                    GB_NEXT (M) ;
                }
                else
                { 
                    // M(iA,j) not present, implicitly false
                    mij = false ;
                }
                // negate the mask M since Mask_comp is true
                mij = !mij ;

                //--------------------------------------------------------------
                // find C(iC,jC), but only if M(iA,j) allows it
                //--------------------------------------------------------------

                if (mij)
                { 

                    //----------------------------------------------------------
                    // C(iC,jC) += scalar
                    //----------------------------------------------------------

                    // direct lookup of C(iC,jC)
                    GB_CDENSE_I_LOOKUP ;

                    // ----[C A 1] or [X A 1]-----------------------------------
                    // [C A 1]: action: ( =C+A ): apply accum
                    // [X A 1]: action: ( undelete ) zombie lives
                    GB_withaccum_C_A_1_scalar ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // C(:,jC) is sparse; use binary search for C
            //------------------------------------------------------------------

            // for each iA in I [...]:
            for (int64_t iA = 0 ; iA < nI ; iA++)
            {

                //--------------------------------------------------------------
                // find M(iA,j)
                //--------------------------------------------------------------

                bool mij ;
                bool found = (pM < pM_end) && (Mi [pM] == iA) ;
                if (found)
                { 
                    // found it
                    cast_M (&mij, Mx +(pM*msize), 0) ;
                    GB_NEXT (M) ;
                }
                else
                { 
                    // M(iA,j) not present, implicitly false
                    mij = false ;
                }
                // negate the mask M since Mask_comp is true
                mij = !mij ;

                //--------------------------------------------------------------
                // find C(iC,jC), but only if M(iA,j) allows it
                //--------------------------------------------------------------

                if (mij)
                {

                    //----------------------------------------------------------
                    // C(iC,jC) += scalar
                    //----------------------------------------------------------

                    // iC = I [iA] ; or I is a colon expression
                    int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                    // binary search for C(iC,jC) in C(:,jC)
                    GB_iC_BINARY_SEARCH ;

                    if (found)
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =C+A ): apply accum
                        // [X A 1]: action: ( undelete ) zombie lives
                        GB_withaccum_C_A_1_scalar ;
                    }
                    else
                    { 
                        // ----[. A 1]------------------------------------------
                        // action: ( insert )
                        GB_D_A_1_scalar ;
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

