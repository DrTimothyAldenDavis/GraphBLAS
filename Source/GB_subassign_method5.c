//------------------------------------------------------------------------------
// GB_subassign_method5: C(I,J) += A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method5
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
    const GrB_Matrix A,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_A ;
    GB_GET_ACCUM ;

    //--------------------------------------------------------------------------
    // Method 5: C(I,J) += A ; no S
    //--------------------------------------------------------------------------

    // time: O(nnz(A)*log(c)) if C standard. O(nnz(A)) if C dense.
    // +O(anvec*log(cnvec)) if C hyper.

    // GB_accum_mask case: C(:,:) = accum (C(:,:),T)

    // PARALLEL: split A into coarse/fine tasks.  Like GB_ewise_slice, with A
    // but no B.  So no need for log^2(n) time GB_slice_vector.  Just slice A.
    // Use GB_ek_slice.

    GBI_for_each_vector (A)
    {

        //----------------------------------------------------------------------
        // get A(:,j)
        //----------------------------------------------------------------------

        GBI_jth_iteration (j, pA, pA_end) ;

        //----------------------------------------------------------------------
        // C(I,j) += A(:,j)
        //----------------------------------------------------------------------

        // get the C(:,jC) vector where jC = J [j]
        int64_t GB_jC_LOOKUP ;

        if (pC_end - pC_start == cvlen)
        {

            //------------------------------------------------------------------
            // C(:,jC) is dense so binary search of C is not needed
            //------------------------------------------------------------------

            for ( ; pA < pA_end ; pA++)
            { 

                //--------------------------------------------------------------
                // consider the entry A(iA,j)
                //--------------------------------------------------------------

                int64_t iA = Ai [pA] ;

                //--------------------------------------------------------------
                // C(iC,jC) += A(iA,j)
                //--------------------------------------------------------------

                // direct lookup of C(iC,jC)
                GB_CDENSE_I_LOOKUP ;

                // ----[C A 1] or [X A 1]---------------------------------------
                // [C A 1]: action: ( =C+A ): apply accum
                // [X A 1]: action: ( undelete ): zombie lives
                GB_withaccum_C_A_1_matrix ;

            }

        }
        else
        {

            //------------------------------------------------------------------
            // C(:,jC) is sparse; use binary search for C
            //------------------------------------------------------------------

            for ( ; pA < pA_end ; pA++)
            {

                //--------------------------------------------------------------
                // consider the entry A(iA,j)
                //--------------------------------------------------------------

                int64_t iA = Ai [pA] ;

                //--------------------------------------------------------------
                // C(iC,jC) += A(iA,j)
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
                    GB_withaccum_C_A_1_matrix ;
                }
                else
                { 
                    // ----[. A 1]----------------------------------------------
                    // action: ( insert )
                    GB_D_A_1_matrix ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

