//------------------------------------------------------------------------------
// GB_subassign_method6: C(I,J)<#M> += A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method6
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
    const bool Mask_comp,
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_MASK ;
    GB_GET_A ;
    GB_GET_ACCUM ;

    //--------------------------------------------------------------------------
    // Method 6: C(I,J)<#M> += A ; where #M denotes M or !M ; no S
    //--------------------------------------------------------------------------

    // time: O(nnz(A)*(log(c)+log(md))+mnvec) if C standard and not
    // dense, where md = avg nnz M(:,j).  O(nnz(A)*(log(md))+mnvec)
    // if C dense.  +O(anvec*log(cnvec)) if C hyper.

    // GB_accum_mask case: C(:,:)<M> = accum (C(:,:),T)

    // PARALLEL: split A into coarse/fine tasks, same as method 5,
    // but also slice M.
    GBI2_for_each_vector (A, M)
    {

        //----------------------------------------------------------------------
        // get A(:,j) and M(:,j)
        //----------------------------------------------------------------------

        GBI2_jth_iteration (Iter, j, pA, pA_end, pM_start, pM_end) ;

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
                // find M(iA,j)
                //--------------------------------------------------------------

                // FUTURE:: skip binary search if M(:,j) dense

                bool mij = true ;
                int64_t pM     = pM_start ;
                int64_t pright = pM_end - 1 ;
                bool found ;
                GB_BINARY_SEARCH (iA, Mi, pM, pright, found) ;
                if (found)
                { 
                    // found it
                    cast_M (&mij, Mx +(pM*msize), 0) ;
                }
                else
                { 
                    // M(iA,j) not present, implicitly false
                    mij = false ;
                }
                if (Mask_comp)
                { 
                    // negate the mask M if Mask_comp is true
                    mij = !mij ;
                }

                //--------------------------------------------------------------
                // find C(iC,jC), but only if M(iA,j) allows it
                //--------------------------------------------------------------

                if (mij)
                { 

                    //----------------------------------------------------------
                    // C(iC,jC) += A(iA,j)
                    //----------------------------------------------------------

                    // direct lookup of C(iC,jC)
                    GB_CDENSE_I_LOOKUP ;

                    // ----[C A 1] or [X A 1]-----------------------------------
                    // [C A 1]: action: ( =C+A ): apply accum
                    // [X A 1]: action: ( undelete ) zombie live
                    GB_withaccum_C_A_1_matrix ;
                }
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
                // find M(iA,j)
                //--------------------------------------------------------------

                // FUTURE:: skip binary search if M(:,j) dense

                bool mij = true ;
                int64_t pM     = pM_start ;
                int64_t pright = pM_end - 1 ;
                bool found ;
                GB_BINARY_SEARCH (iA, Mi, pM, pright, found) ;
                if (found)
                { 
                    // found it
                    cast_M (&mij, Mx +(pM*msize), 0) ;
                }
                else
                { 
                    // M(iA,j) not present, implicitly false
                    mij = false ;
                }
                if (Mask_comp)
                { 
                    // negate the mask M if Mask_comp is true
                    mij = !mij ;
                }

                //--------------------------------------------------------------
                // find C(iC,jC), but only if M(iA,j) allows it
                //--------------------------------------------------------------

                if (mij)
                {

                    //----------------------------------------------------------
                    // C(iC,jC) += A(iA,j)
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
                        GB_withaccum_C_A_1_matrix ;
                    }
                    else
                    { 
                        // ----[. A 1]------------------------------------------
                        // action: ( insert )
                        GB_D_A_1_matrix ;
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

