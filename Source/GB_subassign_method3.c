//------------------------------------------------------------------------------
// GB_subassign_method3: C(I,J) += scalar ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method3
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
    GB_GET_ACCUM_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 3: C(I,J) += scalar ; no S
    //--------------------------------------------------------------------------

    // time: O(nI*nJ*log(c)) if C standard.  O(nI*nJ) if C dense.
    // +O(nJ*log(cnvec)) if C hypersparse.

    // PARALLEL: split of nI*nJ into either all-coarse or all-fine
    // tasks.  All-coarse tasks preferred, but if nJ < nthreads,
    // this does not give enough parallelism.  If split into fine
    // tasks, each fine task needs to know its slice of C(:,jC).

    for (int64_t j = 0 ; j < nJ ; j++)
    {
        // get the C(:,jC) vector where jC = J [j]
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
                // C(iC,jC) += scalar
                //--------------------------------------------------------------

                // direct lookup of C(iC,jC)
                GB_CDENSE_I_LOOKUP ;

                // ----[C A 1] or [X A 1]---------------------------------------
                // [C A 1]: action: ( =C+A ): apply accum
                // [X A 1]: action: ( undelete ): bring zombie back
                GB_withaccum_C_A_1_scalar ;
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
                // C(iC,jC) += scalar
                //--------------------------------------------------------------

                // iC = I [iA] ; or I is a colon expression
                int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                // binary search for C(iC,jC) in C(:,jC)
                GB_iC_BINARY_SEARCH ;

                if (found)
                { 
                    // ----[C A 1] or [X A 1]-----------------------------------
                    // [C A 1]: action: ( =C+A ): apply accum
                    // [X A 1]: action: ( undelete ): zombie lives
                    GB_withaccum_C_A_1_scalar ;
                }
                else
                { 
                    // ----[. A 1]----------------------------------------------
                    // action: ( insert )
                    GB_D_A_1_scalar ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

