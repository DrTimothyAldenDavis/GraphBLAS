//------------------------------------------------------------------------------
// GB_subassign_method9: C(I,J) = A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

GrB_Info GB_subassign_method9
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
    const GrB_Matrix A,
    const GrB_Matrix S,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_A ;
    GB_GET_S ;
    GrB_BinaryOp accum = NULL ;

    //--------------------------------------------------------------------------
    // Method 9: C(I,J) = A ; using S
    //--------------------------------------------------------------------------

    // time:  O(nnz(S)+nnz(A)+nvec(S+A))

    // PARALLEL: split S+A into coarse/fine tasks,
    // same as GB_ewise_slice; no mask.  No need to slice C.
    GBI2_for_each_vector (S, A)
    {

        //----------------------------------------------------------------------
        // get S(:,j) and A(:,j)
        //----------------------------------------------------------------------

        GBI2_jth_iteration (Iter, j, pS, pS_end, pA, pA_end) ;

        //----------------------------------------------------------------------
        // do a 2-way merge of S(:,j) and A(:,j)
        //----------------------------------------------------------------------

        // jC = J [j] ; or J is a colon expression
        int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

        // while both list S (:,j) and A (:,j) have entries
        while (pS < pS_end && pA < pA_end)
        {
            int64_t iS = Si [pS] ;
            int64_t iA = Ai [pA] ;

            if (iS < iA)
            { 
                // ----[C . 1] or [X . 1]---------------------------------------
                // S (i,j) is present but A (i,j) is not
                // [C . 1]: action: ( delete ): becomes a zombie
                // [X . 1]: action: ( X ): still a zombie
                GB_C_S_LOOKUP ;
                GB_noaccum_C_D_1_matrix ;
                GB_NEXT (S) ;

            }
            else if (iA < iS)
            { 
                // ----[. A 1]--------------------------------------------------
                // S (i,j) is not present, A (i,j) is present
                // [. A 1]: action: ( insert )
                // iC = I [iA] ; or I is a colon expression
                int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                GB_D_A_1_matrix ;
                GB_NEXT (A) ;
            }
            else
            { 
                // ----[C A 1] or [X A 1]---------------------------------------
                // both S (i,j) and A (i,j) present
                // [C A 1]: action: ( =A ): copy A into C, no accum
                // [X A 1]: action: ( undelete ): bring zombie back
                GB_C_S_LOOKUP ;
                GB_noaccum_C_A_1_matrix ;
                GB_NEXT (S) ;
                GB_NEXT (A) ;
            }
        }

        // while list S (:,j) has entries.  List A (:,j) exhausted
        while (pS < pS_end)
        { 
            // ----[C . 1] or [X . 1]-------------------------------------------
            // S (i,j) is present but A (i,j) is not
            // [C . 1]: action: ( delete ): becomes a zombie
            // [X . 1]: action: ( X ): still a zombie
            GB_C_S_LOOKUP ;
            GB_noaccum_C_D_1_matrix ;
            GB_NEXT (S) ;
        }

        // while list A (:,j) has entries.  List S (:,j) exhausted
        while (pA < pA_end)
        { 
            // ----[. A 1]------------------------------------------------------
            // S (i,j) is not present, A (i,j) is present
            // [. A 1]: action: ( insert )
            int64_t iA = Ai [pA] ;
            // iC = I [iA] ; or I is a colon expression
            int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
            GB_D_A_1_matrix ;
            GB_NEXT (A) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

