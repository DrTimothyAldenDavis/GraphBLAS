//------------------------------------------------------------------------------
// GB_subassign_method10: C(I,J) += A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Method 10: C(I,J) += A ; using S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           matrix
// S:           constructed (see also Method 5)

// Compare with Method 5, which computes the same thing without creating S.

#define GB_FREE_WORK GB_FREE_2_SLICE

#include "GB_subassign.h"

GrB_Info GB_subassign_method10
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
    GB_GET_ACCUM ;

    //--------------------------------------------------------------------------
    // Method 10: C(I,J) += A ; using S
    //--------------------------------------------------------------------------

    // Time: Close to Optimal.  Every entry in A must be visited, and the
    // corresponding entry in S must then be found.  Time for this phase is
    // Omega(nnz(A)), but S has already been constructed, in Omega(nnz(S))
    // time.  If nnz(S) is very high, then Method 5 is used instead.  As a
    // result, this method simply traverses all of A+S (like GB_add for
    // computing A+S), the same as Method 9.  Time taken is O(nnz(A)+nnz(S)).
    // The only difference is that the traversal of A+S can terminate if A is
    // exhausted.  Entries in S but not A do not actually require any work
    // (unlike Method 9, which must visit all entries in A+S).

    // Method 9 and Method 10 are somewhat similar.  They differ on how C is
    // modified when the entry is present in S but not A.

    // Compare with Methods 14b and 14d

    //--------------------------------------------------------------------------
    // Parallel: Z=A+S (Methods 9, 10, 11c, 12c, 13[abcd], 14[abcd])
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_2_SLICE (A, S) ;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies) reduction(&&:ok)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_TASK_DESCRIPTOR ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; task_ok && k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get A(:,j) and S(:,j)
            //------------------------------------------------------------------

            int64_t j = (Zh == NULL) ? k : Zh [k] ;
            GB_GET_MAPPED_VECTOR (pA, pA_end, pA, pA_end, Ap, j, k, Z_to_X) ;
            GB_GET_MAPPED_VECTOR (pS, pS_end, pB, pB_end, Sp, j, k, Z_to_S) ;

            //------------------------------------------------------------------
            // do a 2-way merge of S(:,j) and A(:,j)
            //------------------------------------------------------------------

            // jC = J [j] ; or J is a colon expression
            int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;

            // while both list S (:,j) and A (:,j) have entries
            while (pS < pS_end && pA < pA_end)
            {
                int64_t iS = Si [pS] ;
                int64_t iA = Ai [pA] ;

                if (iS < iA)
                { 
                    // ----[C . 1] or [X . 1]-----------------------------------
                    // S (i,j) is present but A (i,j) is not
                    // [C . 1]: action: ( C ): no change, with accum
                    // [X . 1]: action: ( X ): still a zombie
                    GB_NEXT (S) ;

                }
                else if (iA < iS)
                { 
                    // ----[. A 1]----------------------------------------------
                    // S (i,j) is not present, A (i,j) is present
                    // [. A 1]: action: ( insert )
                    // iC = I [iA] ; or I is a colon expression
                    int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                    GB_D_A_1_matrix ;
                    GB_NEXT (A) ;
                }
                else
                { 
                    // ----[C A 1] or [X A 1]-----------------------------------
                    // both S (i,j) and A (i,j) present
                    // [C A 1]: action: ( =C+A ): apply accum
                    // [X A 1]: action: ( undelete ): bring zombie back
                    GB_C_S_LOOKUP ;
                    GB_withaccum_C_A_1_matrix ;
                    GB_NEXT (S) ;
                    GB_NEXT (A) ;
                }
            }

            if (!task_ok) break ;

            // ignore the remainder of S (:,j)

            // while list A (:,j) has entries.  List S (:,j) exhausted
            while (pA < pA_end)
            { 
                // ----[. A 1]--------------------------------------------------
                // S (i,j) is not present, A (i,j) is present
                // [. A 1]: action: ( insert )
                int64_t iA = Ai [pA] ;
                // iC = I [iA] ; or I is a colon expression
                int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                GB_D_A_1_matrix ;
                GB_NEXT (A) ;
            }
        }

        //----------------------------------------------------------------------
        // log the result of this task
        //----------------------------------------------------------------------

        ok = ok && task_ok ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_WRAPUP ;
}

