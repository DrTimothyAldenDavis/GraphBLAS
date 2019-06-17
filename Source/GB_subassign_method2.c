//------------------------------------------------------------------------------
// GB_subassign_method2: C(I,J)<M> += scalar ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method2
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
    // Method 2: C(I,J)<M> += scalar ; no S
    //--------------------------------------------------------------------------

    // PARALLEL: split M into coarse/fine tasks, or use GB_ek_slice
    GBI_for_each_vector (M)
    {

        //----------------------------------------------------------------------
        // get M(:,j)
        //----------------------------------------------------------------------

        GBI_jth_iteration (j, pM, pM_end) ;

        //----------------------------------------------------------------------
        // C(I,j)<M(:,j)> += scalar
        //----------------------------------------------------------------------

        // get the C(:,jC) vector where jC = J [j]
        int64_t GB_jC_LOOKUP ;

        if (pC_end - pC_start == cvlen)
        {

            //------------------------------------------------------------------
            // C(:,jC) is dense so the binary search of C is not needed
            //------------------------------------------------------------------

            GBI_for_each_entry (j, pM, pM_end)
            {

                //--------------------------------------------------------------
                // consider the entry M(i,j)
                //--------------------------------------------------------------

                bool mij ;
                cast_M (&mij, Mx +(pM*msize), 0) ;

                //--------------------------------------------------------------
                // update C(iC,jC), but only if M(iC,j) allows it
                //--------------------------------------------------------------

                if (mij)
                { 

                    //----------------------------------------------------------
                    // C(iC,jC) += scalar
                    //----------------------------------------------------------

                    int64_t iA = Mi [pM] ;
                    GB_CDENSE_I_LOOKUP ;

                    // ----[C A 1] or [X A 1]-----------------------------------
                    // [C A 1]: action: ( =C+A ): apply accum
                    // [X A 1]: action: ( undelete ): bring zombie back
                    GB_withaccum_C_A_1_scalar ;
                }
            }
        }
        else
        {

            //------------------------------------------------------------------
            // C(:,jC) is sparse; use binary search for C
            //------------------------------------------------------------------

            GBI_for_each_entry (j, pM, pM_end)
            {

                //--------------------------------------------------------------
                // consider the entry M(i,j)
                //--------------------------------------------------------------

                bool mij ;
                cast_M (&mij, Mx +(pM*msize), 0) ;

                //--------------------------------------------------------------
                // find C(iC,jC), but only if M(i,j) allows it
                //--------------------------------------------------------------

                if (mij)
                {

                    //----------------------------------------------------------
                    // C(iC,jC) += scalar
                    //----------------------------------------------------------

                    // binary search for C(iC,jC) in C(:,jC)
                    int64_t iA = Mi [pM] ;
                    int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                    GB_iC_BINARY_SEARCH ;

                    if (found)
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =C+A ): apply accum
                        // [X A 1]: action: ( undelete ): zombie lives
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

