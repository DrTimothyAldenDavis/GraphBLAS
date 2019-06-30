//------------------------------------------------------------------------------
// GB_subassign_method15: C(I,J)<M> = A ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Method 15: C(I,J)<M> = A ; no S

// M:           present
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           matrix
// S:           none (see also Method 13d)

#include "GB_subassign.h"

GrB_Info GB_subassign_method15
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
    GrB_BinaryOp accum = NULL ;

    //--------------------------------------------------------------------------
    // Method 15: C(I,J)<M> = A ; no S
    //--------------------------------------------------------------------------

    // Time: O(nnz(M)*(log(a)+log(c)), where a and c are the # of entries in a
    // vector of A and C, respectively.  The entries in the intersection of M
    // (where the entries are true) and the matrix addition C(I,J)+A must be
    // examined.  This method scans M, and searches for entries in A and C(I,J)
    // using two binary searches.  If M is very dense, this method can be
    // slower than Method 13d.

    //--------------------------------------------------------------------------
    // Parallel: slice M into coarse/fine tasks (Method 1, 2, 5, 6a, 15)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_1_SLICE (M) ;

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
            // get j, the kth vector of M
            //------------------------------------------------------------------

            int64_t j = (Mh == NULL) ? k : Mh [k] ;
            GB_GET_VECTOR (pM, pM_end, pA, pA_end, Mp, k) ;
            int64_t mjnz = pM_end - pM ;
            if (mjnz == 0) continue ;

            //------------------------------------------------------------------
            // get A(:,j)
            //------------------------------------------------------------------

            int64_t pA, pA_end ;
            GB_VECTOR_LOOKUP (pA, pA_end, A, j) ;
            int64_t ajnz = pA_end - pA ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;
            int64_t cjnz = pC_end - pC_start ;
            if (cjnz == 0 && ajnz == 0) continue ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> = A(:,j)
            //------------------------------------------------------------------

            for ( ; pM < pM_end ; pM++)
            {

                //----------------------------------------------------------
                // consider the entry M(iA,j)
                //----------------------------------------------------------

                bool mij ;
                cast_M (&mij, Mx +(pM*msize), 0) ;

                //----------------------------------------------------------
                // update C(iC,jC), but only if M(iA,j) allows it
                //----------------------------------------------------------

                if (mij)
                {
                    int64_t iA = Mi [pM] ;

                    // find iA in A(:,j)
                    int64_t apright = pA_end - 1 ;
                    bool aij_found ;
                    GB_BINARY_SEARCH (iA, Ai, pA, apright, aij_found) ;

                    // find iC in C(:,jC)
                    GB_iC_BINARY_SEARCH ;
                    bool cij_found = found ;

                    if (cij_found && !aij_found)
                    { 
                        // C (iC,jC) is present but A (i,j) is not
                        // ----[C . 1] or [X . 1]-------------------------------
                        // [C . 1]: action: ( delete ): becomes zombie
                        // [X . 1]: action: ( X ): still zombie
//                      printf ("delete ("GBd":"GBd") = ("GBd":"GBd")\n",
//                          iC, jC, iA, j) ;
                        GB_DELETE_ENTRY ;
                    }
                    else if (!cij_found && aij_found)
                    { 
                        // C (iC,jC) is not present, A (i,j) is present
                        // ----[. A 1]------------------------------------------
                        // [. A 1]: action: ( insert )
//                      printf (GBd" pending ("GBd":"GBd") = ("GBd":"GBd")\n",
//                          task_pending, iC, jC, iA, j) ;
                        task_pending++ ;
                    }
                    else if (cij_found && aij_found)
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =A ): A to C no accum
                        // [X A 1]: action: ( undelete ): zombie lives
//                      printf ("copy ("GBd":"GBd") = ("GBd":"GBd")\n",
//                          iC, jC, iA, j) ;
                        GB_noaccum_C_A_1_matrix ;
                    }
                }
            }
        }

        GB_PHASE1_TASK_WRAPUP ;
    }

    //--------------------------------------------------------------------------
    // phase 2: insert pending tuples
    //--------------------------------------------------------------------------

//    printf ("\n") ;

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
            // get j, the kth vector of M
            //------------------------------------------------------------------

            int64_t j = (Mh == NULL) ? k : Mh [k] ;
            GB_GET_VECTOR (pM, pM_end, pA, pA_end, Mp, k) ;
            int64_t mjnz = pM_end - pM ;
            if (mjnz == 0) continue ;

            //------------------------------------------------------------------
            // get A(:,j)
            //------------------------------------------------------------------

            int64_t pA, pA_end ;
            GB_VECTOR_LOOKUP (pA, pA_end, A, j) ;
            int64_t ajnz = pA_end - pA ;
            if (ajnz == 0) continue ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> = A(:,j)
            //------------------------------------------------------------------

            for ( ; pM < pM_end ; pM++)
            {

                //----------------------------------------------------------
                // consider the entry M(iA,j)
                //----------------------------------------------------------

                bool mij ;
                cast_M (&mij, Mx +(pM*msize), 0) ;

                //----------------------------------------------------------
                // update C(iC,jC), but only if M(iA,j) allows it
                //----------------------------------------------------------

                if (mij)
                { 
                    int64_t iA = Mi [pM] ;

                    // find iA in A(:,j)
                    int64_t apright = pA_end - 1 ;
                    bool aij_found ;
                    GB_BINARY_SEARCH (iA, Ai, pA, apright, aij_found) ;

                    // find iC in C(:,jC)
                    GB_iC_BINARY_SEARCH ;
                    bool cij_found = found ;

                    if (!cij_found && aij_found)
                    { 
                        // C (iC,jC) is not present, A (i,j) is present
                        // ----[. A 1]--------------------------------------
                        // [. A 1]: action: ( insert )
//                      printf (GBd" pending ("GBd":"GBd") = ("GBd":"GBd
//                          ")\n", task_pending, iC, jC, iA, j) ;
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

