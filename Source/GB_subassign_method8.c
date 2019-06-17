//------------------------------------------------------------------------------
// GB_subassign_method8: C(I,J) += scalar ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method8
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
    GB_GET_S ;
    GB_GET_ACCUM_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 8: C(I,J) += scalar ; using S
    //--------------------------------------------------------------------------

    // time: O(nI*nJ)

    // PARALLEL: split nI*nJ into all-coarse or all-fine tasks.
    // same as method 7
    GBI2s_for_each_vector (S, scalar)
    {

        //----------------------------------------------------------------------
        // get S(:,j) and the scalar
        //----------------------------------------------------------------------

        GBI2s_jth_iteration (Iter, j, pS, pS_end) ;

        //----------------------------------------------------------------------
        // do a 2-way merge of S(:,j) and the scalar
        //----------------------------------------------------------------------

        // jC = J [j] ; or J is a colon expression
        int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

        // for each iA in I [...]:
        for (int64_t iA = 0 ; iA < nI ; iA++)
        {
            bool found = (pS < pS_end) && (Si [pS] == iA) ;

            if (!found)
            { 
                // ----[. A 1]--------------------------------------------------
                // S (i,j) is not present, the scalar is present
                // [. A 1]: action: ( insert )
                // iC = I [iA] ; or I is a colon expression
                int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                GB_D_A_1_scalar ;
            }
            else
            { 
                // ----[C A 1] or [X A 1]---------------------------------------
                // both S (i,j) and A (i,j) present
                // [C A 1]: action: ( =C+A ): apply accum
                // [X A 1]: action: ( undelete ): zombie lives
                GB_C_S_LOOKUP ;
                GB_withaccum_C_A_1_scalar ;
                GB_NEXT (S) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}
