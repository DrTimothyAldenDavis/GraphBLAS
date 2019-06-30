//------------------------------------------------------------------------------
// GB_subassign_method5: C(I,J) += A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Method 5: C(I,J) += A ; no S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           matrix
// S:           none (see also Method 10)

// Compare with Method 10, which computes the same thing, but creates S first.

// Both methods are Omega(nnz(A)), since all entries in A must be considered,
// and inserted or accumulated into C.  Method 5 uses a binary search to find
// the corresponding entry in C, for each entry in A, and thus takes
// O(nnz(A)*log(c)) time in general, if c is the # entries in a given vector of
// C.  Method 10 takes O(nnz(A)+nnz(S)), plus any additional time required to
// search C to construct S.  If nnz(A) << nnz (S), then Method 10 is costly,
// and Method 5 is used instead.

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

    // Time: Close to Optimal.  All entries in A must be examined, taking
    // Omega(nnz(A)) time.  This method then finds all corresponding entries in
    // C, and updates them.  If the entry is not present in C, it is inserted.
    // This binary search adds a log(n) factor to the time, and thus the total
    // time is O(nnz(A)*log(n)), or O(nnz(A)*log(c)) if c is the largest number
    // of entries in any vector of C.  An additional time of
    // O(anvec*log(Cnvec)) is added if C is hypersparse.

    //--------------------------------------------------------------------------
    // Parallel: slice A into coarse/fine tasks (Method 1, 2, 5, 6a, 15)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_1_SLICE (A) ;

    //--------------------------------------------------------------------------
    // phase 1: create zombies, update entries, and count pending tuples
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_TASK_DESCRIPTOR ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get j, the kth vector of A
            //------------------------------------------------------------------

            int64_t j = (Ah == NULL) ? k : Ah [k] ;
            GB_GET_VECTOR (pA, pA_end, pA, pA_end, Ap, k) ;
            int64_t ajnz = pA_end - pA ;
            if (ajnz == 0) continue ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;

            //------------------------------------------------------------------
            // C(I,jC) += A(:,j)
            //------------------------------------------------------------------

            if (pC_end - pC_start == cvlen)
            {

                //--------------------------------------------------------------
                // C(:,jC) is dense so binary search of C is not needed
                //--------------------------------------------------------------

                for ( ; pA < pA_end ; pA++)
                { 

                    //----------------------------------------------------------
                    // consider the entry A(iA,j)
                    //----------------------------------------------------------

                    int64_t iA = Ai [pA] ;

                    //----------------------------------------------------------
                    // C(iC,jC) += A(iA,j)
                    //----------------------------------------------------------

                    // direct lookup of C(iC,jC)
                    GB_iC_DENSE_LOOKUP ;

                    // ----[C A 1] or [X A 1]-----------------------------------
                    // [C A 1]: action: ( =C+A ): apply accum
                    // [X A 1]: action: ( undelete ): zombie lives
                    GB_withaccum_C_A_1_matrix ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse; use binary search for C
                //--------------------------------------------------------------

                for ( ; pA < pA_end ; pA++)
                {

                    //----------------------------------------------------------
                    // consider the entry A(iA,j)
                    //----------------------------------------------------------

                    int64_t iA = Ai [pA] ;

                    //----------------------------------------------------------
                    // C(iC,jC) += A(iA,j)
                    //----------------------------------------------------------

                    // binary search for C(iC,jC) in C(:,jC)
                    GB_iC_BINARY_SEARCH ;

                    if (found)
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =C+A ): apply accum
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_withaccum_C_A_1_matrix ;
                    }
                    else
                    { 
                        // ----[. A 1]------------------------------------------
                        // action: ( insert )
                        task_pending++ ;
                    }
                }
            }
        }

        GB_PHASE1_TASK_WRAPUP ;
    }

    //--------------------------------------------------------------------------
    // phase 2: insert pending tuples
    //--------------------------------------------------------------------------

    GB_PENDING_CUMSUM ;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(&&:pending_sorted)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_TASK_DESCRIPTOR ;
        GB_START_PENDING_INSERTION ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get j, the kth vector of A
            //------------------------------------------------------------------

            int64_t j = (Ah == NULL) ? k : Ah [k] ;
            GB_GET_VECTOR (pA, pA_end, pA, pA_end, Ap, k) ;
            int64_t ajnz = pA_end - pA ;
            if (ajnz == 0) continue ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;

            //------------------------------------------------------------------
            // C(I,jC) += A(:,j)
            //------------------------------------------------------------------

            if (pC_end - pC_start != cvlen)
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse; use binary search for C
                //--------------------------------------------------------------

                for ( ; pA < pA_end ; pA++)
                {

                    //----------------------------------------------------------
                    // consider the entry A(iA,j)
                    //----------------------------------------------------------

                    int64_t iA = Ai [pA] ;

                    //----------------------------------------------------------
                    // C(iC,jC) += A(iA,j)
                    //----------------------------------------------------------

                    // binary search for C(iC,jC) in C(:,jC)
                    GB_iC_BINARY_SEARCH ;

                    if (!found)
                    { 
                        // ----[. A 1]------------------------------------------
                        // action: ( insert )
                        GB_PENDING_INSERT (Ax +(pA*asize)) ;
                    }
                }
            }
        }

        GB_PHASE2_TASK_WRAPUP ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_WRAPUP ;
}

