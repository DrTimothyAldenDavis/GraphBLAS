//------------------------------------------------------------------------------
// GB_subassign_method9: C(I,J) = A ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Method 9: C(I,J) = A ; using S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           matrix
// S:           constructed

#define GB_FREE_WORK GB_FREE_2_SLICE

#include "GB_subassign.h"

GrB_Info GB_subassign_method9
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

    // Time: Optimal.  All entries in A+S must be examined, so the work is
    // Omega (nnz(A)+nnz(S)).

    // Method 9 and Method 10 are somewhat similar.  They differ on how C is
    // modified when the entry is present in S but not A.

    //--------------------------------------------------------------------------
    // Parallel: Z=A+S (Methods 9, 10, 11c, 12c, 13[abcd], 14[abcd])
    //--------------------------------------------------------------------------

// double t = omp_get_wtime ( ) ;
    GB_SUBASSIGN_2_SLICE (A, S) ;
// t = omp_get_wtime ( ) - t ; printf ("schedule time %g\n", t) ;
// t = omp_get_wtime ( ) ;

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
                    // [C . 1]: action: ( delete ): becomes zombie
                    // [X . 1]: action: ( X ): still a zombie
                    GB_C_S_LOOKUP ;
                    GB_DELETE_ENTRY ;
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
                    // [C A 1]: action: ( =A ): copy A into C, no accum
                    // [X A 1]: action: ( undelete ): bring zombie back
                    GB_C_S_LOOKUP ;
                    GB_noaccum_C_A_1_matrix ;
                    GB_NEXT (S) ;
                    GB_NEXT (A) ;
                }
            }

            if (!task_ok) break ;

            // while list S (:,j) has entries.  List A (:,j) exhausted
            while (pS < pS_end)
            { 
                // ----[C . 1] or [X . 1]---------------------------------------
                // S (i,j) is present but A (i,j) is not
                // [C . 1]: action: ( delete ): becomes zombie
                // [X . 1]: action: ( X ): still a zombie
                GB_C_S_LOOKUP ;
                GB_DELETE_ENTRY ;
                GB_NEXT (S) ;
            }

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

// t = omp_get_wtime ( ) - t ; printf ("task time %g\n", t) ;
// t = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_WRAPUP ;

//  if (ok)
//  {
//      /* finalize the zombie count and merge all pending tuples */
//      C->nzombies = nzombies ;
//      ok = GB_Pending_merge (&(C->Pending), atype, accum, is_matrix,
//          TaskList, ntasks, nthreads) ;
// t = omp_get_wtime ( ) - t ; printf ("merge time %g\n", t) ;
//  }
//    GB_FREE_ALL ;
//    return (ok ? GrB_SUCCESS : GB_OUT_OF_MEMORY) ;

}

