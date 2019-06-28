//------------------------------------------------------------------------------
// GB_subassign_method3: C(I,J) += scalar ; no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Method 3: C(I,J) += scalar ; no S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       present
// A:           scalar
// S:           none (see also Method 8)

// Compare with Method 8, which computes the same thing, but creates S first.

#include "GB_subassign.h"

GrB_Info GB_subassign_method3
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
    const void *scalar,
    const GrB_Type atype,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_ACCUM_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 3: C(I,J) += scalar ; no S
    //--------------------------------------------------------------------------

    // Time: Close to optimal; must visit all IxJ, so Omega(|I|*|J|) is
    // required.

    // The submatrix C(I,J) becomes completely dense.  Existing entries in
    // C(I,J) are found and updated, adding an extra log(n) term to the time
    // for this method.  Total time is thus O(|I|*|J|*log(n)) in the worst
    // case, plus O(|J|*log(Cnvec)) if C is hypersparse.

    // Method 8 computes the same thing as Method 3, but searches for the
    // entries in C by constructing S first.

    //--------------------------------------------------------------------------
    // Parallel: all IxJ (Methods 3, 4, 7, 8, 11a, 11b, 12a, 12b)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_IXJ_SLICE (C) ;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies) reduction(&&:ok)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_GET_IXJ_TASK_DESCRIPTOR ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t j = kfirst ; task_ok && j <= klast ; j++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            GB_GET_jC ;

            if (pC_end - pC_start == cvlen)
            {

                //--------------------------------------------------------------
                // C(:,jC) is dense so binary search of C is not needed
                //--------------------------------------------------------------

                for (int64_t iA = iA_start ; iA < iA_end ; iA++)
                { 

                    //----------------------------------------------------------
                    // C(iC,jC) += scalar
                    //----------------------------------------------------------

                    // direct lookup of C(iC,jC)
                    GB_iC_DENSE_LOOKUP ;

                    // ----[C A 1] or [X A 1]-----------------------------------
                    // [C A 1]: action: ( =C+A ): apply accum
                    // [X A 1]: action: ( undelete ): bring zombie back
                    GB_withaccum_C_A_1_scalar ;
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
                    // C(iC,jC) += scalar
                    //----------------------------------------------------------

                    // binary search for C(iC,jC) in C(:,jC)
                    GB_iC_BINARY_SEARCH ;

                    if (found)
                    { 
                        // ----[C A 1] or [X A 1]-------------------------------
                        // [C A 1]: action: ( =C+A ): apply accum
                        // [X A 1]: action: ( undelete ): zombie lives
                        GB_withaccum_C_A_1_scalar ;
                    }
                    else
                    { 
                        // ----[. A 1]------------------------------------------
                        // action: ( insert )
                        GB_D_A_1_scalar ;
                    }
                }
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

