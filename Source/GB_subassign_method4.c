//------------------------------------------------------------------------------
// GB_subassign_method4: C(I,J)<!M> += scalar ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Method 4: C(I,J)<!M> += scalar ; no S

// M:           present
// Mask_comp:   true
// C_replace:   false
// accum:       present
// A:           scalar
// S:           none (see also Method 12b)

// Compare with Method 12b, which computes the same thing, but creates S first.

#include "GB_subassign.h"

GrB_Info GB_subassign_method4
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
    // Method 4: C(I,J)<!M> += scalar ; no S
    //--------------------------------------------------------------------------

    // Time: Close to optimal; must visit all IxJ, so Omega(|I|*|J|) is
    // required.  The sparsity of !M cannot be exploited.

    // The submatrix C(I,J) becomes completely dense.  Existing entries in
    // C(I,J) are found and updated, adding an extra log(n) term to the time
    // for this method.  Total time is thus O(|I|*|J|*log(n)) in the worst
    // case, plus O(|J|*log(Cnvec)) if C is hypersparse.

    // Method 12b computes the same thing as Method 4, but searches for the
    // entries in C by constructing S first.

    // The mask !M cannot be easily exploited.  The traversal of M is identical
    // to the traversal of S in Method 8.

    //--------------------------------------------------------------------------
    // Parallel: all IxJ (Methods 3, 4, 7, 8, 11a, 11b, 12a, 12b)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_IXJ_SLICE (C) ;

    // Each task must also look up its part of M, but this does not affect
    // the parallel tasks.  Total work is about the same as Method 3.

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

        GB_GET_IXJ_TASK_DESCRIPTOR ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t j = kfirst ; j <= klast ; j++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;

            //------------------------------------------------------------------
            // find M(iA_start,j)
            //------------------------------------------------------------------

            GB_GET_VECTOR_FOR_IXJ (M) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC)<!M> = scalar
            //------------------------------------------------------------------

            if (pC_end - pC_start == cvlen)
            {

                //--------------------------------------------------------------
                // C(:,jC) is dense so binary search of C is not needed
                //--------------------------------------------------------------

                for (int64_t iA = iA_start ; iA < iA_end ; iA++)
                {

                    //----------------------------------------------------------
                    // find M(iA,j)
                    //----------------------------------------------------------

                    bool mij ;
                    bool found = (pM < pM_end) && (Mi [pM] == iA) ;
                    if (found)
                    { 
                        // found it
                        cast_M (&mij, Mx +(pM*msize), 0) ;
                        GB_NEXT (M) ;
                    }
                    else
                    { 
                        // M(iA,j) not present, implicitly false
                        mij = false ;
                    }

                    // complement the mask entry mij since Mask_comp is true
                    mij = !mij ;

                    //----------------------------------------------------------
                    // find C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (mij)
                    { 

                        //------------------------------------------------------
                        // C(iC,jC) += scalar
                        //------------------------------------------------------

                        // direct lookup of C(iC,jC)
                        GB_iC_DENSE_LOOKUP ;

                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =C+A ): apply accum
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_withaccum_C_A_1_scalar ;
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse; use binary search for C
                //--------------------------------------------------------------

                for (int64_t iA = iA_start ; iA < iA_end ; iA++)
                {

                    //----------------------------------------------------------
                    // find M(iA,j)
                    //----------------------------------------------------------

                    bool mij ;
                    bool found = (pM < pM_end) && (Mi [pM] == iA) ;
                    if (found)
                    { 
                        // found it
                        cast_M (&mij, Mx +(pM*msize), 0) ;
                        GB_NEXT (M) ;
                    }
                    else
                    { 
                        // M(iA,j) not present, implicitly false
                        mij = false ;
                    }

                    // complement the mask entry mij since Mask_comp is true
                    mij = !mij ;

                    //----------------------------------------------------------
                    // find C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (mij)
                    {

                        //------------------------------------------------------
                        // C(iC,jC) += scalar
                        //------------------------------------------------------

                        // binary search for C(iC,jC) in C(:,jC)
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

        GB_GET_IXJ_TASK_DESCRIPTOR ;
        GB_START_PENDING_INSERTION ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t j = kfirst ; j <= klast ; j++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;

            //------------------------------------------------------------------
            // find M(iA_start,j)
            //------------------------------------------------------------------

            GB_GET_VECTOR_FOR_IXJ (M) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC)<!M> = scalar
            //------------------------------------------------------------------

            if (pC_end - pC_start != cvlen)
            {

                //--------------------------------------------------------------
                // C(:,jC) is sparse; use binary search for C
                //--------------------------------------------------------------

                for (int64_t iA = iA_start ; iA < iA_end ; iA++)
                {

                    //----------------------------------------------------------
                    // find M(iA,j)
                    //----------------------------------------------------------

                    bool mij ;
                    bool found = (pM < pM_end) && (Mi [pM] == iA) ;
                    if (found)
                    { 
                        // found it
                        cast_M (&mij, Mx +(pM*msize), 0) ;
                        GB_NEXT (M) ;
                    }
                    else
                    { 
                        // M(iA,j) not present, implicitly false
                        mij = false ;
                    }

                    // complement the mask entry mij since Mask_comp is true
                    mij = !mij ;

                    //----------------------------------------------------------
                    // find C(iC,jC), but only if M(iA,j) allows it
                    //----------------------------------------------------------

                    if (mij)
                    {

                        //------------------------------------------------------
                        // C(iC,jC) += scalar
                        //------------------------------------------------------

                        // binary search for C(iC,jC) in C(:,jC)
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

