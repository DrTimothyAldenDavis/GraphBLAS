//------------------------------------------------------------------------------
// GB_subassign_23_template: C += A where C is dense; A is sparse or dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_unused.h"

{

    //--------------------------------------------------------------------------
    // get C and A
    //--------------------------------------------------------------------------

    ASSERT (!C->iso) ;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const bool A_iso = A->iso ;
    GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    ASSERT (GB_is_dense (C)) ;
    const int64_t cnz = GB_nnz_held (C) ;
    #ifndef GB_GENERIC
    GB_Y_TYPE ywork ;
    if (A_iso)
    {
        // ywork = (ytype) Ax [0]
        GB_COPY_aij_to_ywork (ywork, Ax, 0, true) ;
    }
    #endif

    if (GB_IS_BITMAP (A))
    {

        //----------------------------------------------------------------------
        // C += A when C is dense and A is bitmap
        //----------------------------------------------------------------------

        const int8_t *restrict Ab = A->b ;
        int64_t p ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            if (!Ab [p]) continue ;
            // Cx [p] += (ytype) Ax [p], with typecasting
            GB_ACCUMULATE_aij (Cx, p, Ax, p, A_iso, ywork) ;
            // GB_COPY_aij_to_ywork (ywork, Ax, p, A_iso) ;
            // GB_ACCUMULATE_scalar (Cx, p, ywork) ;
        }

    }
    else if (A_ek_slicing == NULL)
    {

        //----------------------------------------------------------------------
        // C += A when both C and A are dense
        //----------------------------------------------------------------------

        ASSERT (GB_is_dense (A)) ;
        int64_t p ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            // Cx [p] += (ytype) Ax [p], with typecasting
            GB_ACCUMULATE_aij (Cx, p, Ax, p, A_iso, ywork) ;
            // GB_COPY_aij_to_ywork (ywork, Ax, p, A_iso) ;
            // GB_ACCUMULATE_scalar (Cx, p, ywork) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C += A when C is dense and A is sparse
        //----------------------------------------------------------------------

        ASSERT (GB_JUMBLED_OK (A)) ;

        const int64_t *restrict Ap = A->p ;
        const int64_t *restrict Ah = A->h ;
        const int64_t *restrict Ai = A->i ;
        const int64_t avlen = A->vlen ;
        const int64_t cvlen = C->vlen ;
        bool A_jumbled = A->jumbled ;

        const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
        const int64_t *restrict klast_Aslice  = kfirst_Aslice + A_ntasks ;
        const int64_t *restrict pstart_Aslice = klast_Aslice + A_ntasks ;

        int taskid ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
        for (taskid = 0 ; taskid < A_ntasks ; taskid++)
        {

            // if kfirst > klast then taskid does no work at all
            int64_t kfirst = kfirst_Aslice [taskid] ;
            int64_t klast  = klast_Aslice  [taskid] ;

            //------------------------------------------------------------------
            // C(:,kfirst:klast) += A(:,kfirst:klast)
            //------------------------------------------------------------------

            for (int64_t k = kfirst ; k <= klast ; k++)
            {

                //--------------------------------------------------------------
                // find the part of A(:,k) and C(:,k) for this task
                //--------------------------------------------------------------

                int64_t j = GBH_A (Ah, k) ;
                int64_t my_pA_start, my_pA_end ;
                GB_get_pA (&my_pA_start, &my_pA_end, taskid, k,
                    kfirst, klast, pstart_Aslice, Ap, avlen) ;

                int64_t pA_start = GBP_A (Ap, k, avlen) ;
                int64_t pA_end   = GBP_A (Ap, k+1, avlen) ;
                bool ajdense = ((pA_end - pA_start) == cvlen) ;

                // pC points to the start of C(:,j)
                int64_t pC = j * cvlen ;

                //--------------------------------------------------------------
                // C(:,j) += A(:,j)
                //--------------------------------------------------------------

                if (ajdense && !A_jumbled)
                {

                    //----------------------------------------------------------
                    // both C(:,j) and A(:,j) are dense
                    //----------------------------------------------------------

                    GB_PRAGMA_SIMD_VECTORIZE
                    for (int64_t pA = my_pA_start ; pA < my_pA_end ; pA++)
                    { 
                        int64_t i = pA - pA_start ;
                        int64_t p = pC + i ;
                        // Cx [p] += (ytype) Ax [pA], with typecasting
                        GB_ACCUMULATE_aij (Cx, p, Ax, pA, A_iso, ywork) ;
                        // GB_COPY_aij_to_ywork (ywork, Ax, pA, A_iso) ;
                        // GB_ACCUMULATE_scalar (Cx, p, ywork) ;
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // C(:,j) is dense; A(:,j) is sparse 
                    //----------------------------------------------------------

                    GB_PRAGMA_SIMD_VECTORIZE
                    for (int64_t pA = my_pA_start ; pA < my_pA_end ; pA++)
                    { 
                        int64_t i = Ai [pA] ;
                        int64_t p = pC + i ;
                        // Cx [p] += (ytype) Ax [pA], with typecasting
                        GB_ACCUMULATE_aij (Cx, p, Ax, pA, A_iso, ywork) ;
                        // GB_COPY_aij_to_ywork (ywork, Ax, pA, A_iso) ;
                        // GB_ACCUMULATE_scalar (Cx, p, ywork) ;
                    }
                }
            }
        }
    }
}

