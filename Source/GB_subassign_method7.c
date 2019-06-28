//------------------------------------------------------------------------------
// GB_subassign_method7: C(I,J) = scalar ; using S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Method 7: C(I,J) = scalar ; using S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           constructed

#include "GB_subassign.h"

GrB_Info GB_subassign_method7
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
    const void *scalar,
    const GrB_Type atype,
    const GrB_Matrix S,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C ;
    GB_GET_SCALAR ;
    GB_GET_S ;
    GrB_BinaryOp accum = NULL ;

    //--------------------------------------------------------------------------
    // Method 7: C(I,J) = scalar ; using S
    //--------------------------------------------------------------------------

    // Time: Optimal; must visit all IxJ, so Omega(|I|*|J|) is required.

    // Entries in S are found and the corresponding entry in C replaced with
    // the scalar.  The traversal of S is identical to the traversal of M in
    // Method 4.

    // Method 7 and Method 8 are very similar.

    //--------------------------------------------------------------------------
    // Parallel: all IxJ (Methods 3, 4, 7, 8, 11a, 11b, 12a, 12b)
    //--------------------------------------------------------------------------

    GB_SUBASSIGN_IXJ_SLICE (NULL) ;

    // Each task must also look up its part of S, but this does not affect
    // the parallel tasks.  Total work is about the same as Method 3.

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

            //------------------------------------------------------------------
            // get S(iA_start:end,j)
            //------------------------------------------------------------------

            GB_GET_VECTOR_FOR_IXJ (S) ;

            //------------------------------------------------------------------
            // C(I(iA_start,iA_end-1),jC) = scalar
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            {
                bool found = (pS < pS_end) && (Si [pS] == iA) ;
                if (!found)
                { 
                    // ----[. A 1]----------------------------------------------
                    // S (i,j) is not present, the scalar is present
                    // [. A 1]: action: ( insert )
                    // iC = I [iA] ; or I is a colon expression
                    int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                    GB_D_A_1_scalar ;
                }
                else
                { 
                    // ----[C A 1] or [X A 1]-----------------------------------
                    // both S (i,j) and A (i,j) present
                    // [C A 1]: action: ( =A ): scalar to C, no accum
                    // [X A 1]: action: ( undelete ): zombie lives
                    GB_C_S_LOOKUP ;
                    GB_noaccum_C_A_1_scalar ;
                    GB_NEXT (S) ;
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

