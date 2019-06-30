//------------------------------------------------------------------------------
// GB_subassign_method2: C(I,J)<M> += scalar ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Method 2: C(I,J)<M> += scalar ; no S

// M:           present
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           scalar
// S:           none

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

    // Time: Close to Optimal:  same as Method 1.

    // Method 1 and Method 2 are very similar.

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
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> += scalar ; no S
            //------------------------------------------------------------------

            if (pC_end - pC_start == cvlen)
            {

                //--------------------------------------------------------------
                // C(:,jC) is dense so the binary search of C is not needed
                //--------------------------------------------------------------

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

                        //------------------------------------------------------
                        // C(iC,jC) += scalar
                        //------------------------------------------------------

                        int64_t iA = Mi [pM] ;
                        GB_iC_DENSE_LOOKUP ;

                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =C+A ): apply accum
                        // [X A 1]: action: ( undelete ): bring zombie back
                        GB_withaccum_C_A_1_scalar ;
                    }
                }
            }
            else
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse; use binary search for C
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // consider the entry M(iA,j)
                    //----------------------------------------------------------

                    bool mij ;
                    cast_M (&mij, Mx +(pM*msize), 0) ;

                    //----------------------------------------------------------
                    // find C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (mij)
                    {

                        //------------------------------------------------------
                        // C(iC,jC) += scalar
                        //------------------------------------------------------

                        // binary search for C(iC,jC) in C(:,jC)
                        int64_t iA = Mi [pM] ;
                        GB_iC_BINARY_SEARCH ;

                        if (found)
                        { 
                            // ----[C A 1] or [X A 1]---------------------------
                            // [C A 1]: action: ( =C+A ): apply accum
                            // [X A 1]: action: ( undelete ): zombie lives
                            GB_withaccum_C_A_1_scalar ;
                        }
                        else
                        { 
                            // ----[. A 1]--------------------------------------
                            // action: ( insert )
                            task_pending++ ;
                        }
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
            // get j, the kth vector of M
            //------------------------------------------------------------------

            int64_t j = (Mh == NULL) ? k : Mh [k] ;
            GB_GET_VECTOR (pM, pM_end, pA, pA_end, Mp, k) ;
            int64_t mjnz = pM_end - pM ;
            if (mjnz == 0) continue ;

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;

            //------------------------------------------------------------------
            // C(I,jC)<M(:,j)> += scalar ; no S
            //------------------------------------------------------------------

            if (pC_end - pC_start != cvlen)
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse; use binary search for C
                //--------------------------------------------------------------

                for ( ; pM < pM_end ; pM++)
                {

                    //----------------------------------------------------------
                    // consider the entry M(iA,j)
                    //----------------------------------------------------------

                    bool mij ;
                    cast_M (&mij, Mx +(pM*msize), 0) ;

                    //----------------------------------------------------------
                    // find C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (mij)
                    {

                        //------------------------------------------------------
                        // C(iC,jC) += scalar
                        //------------------------------------------------------

                        // binary search for C(iC,jC) in C(:,jC)
                        int64_t iA = Mi [pM] ;
                        GB_iC_BINARY_SEARCH ;

                        if (!found)
                        { 
                            // ----[. A 1]--------------------------------------
                            // action: ( insert )
                            GB_PENDING_INSERT (scalar) ;
                        }
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

