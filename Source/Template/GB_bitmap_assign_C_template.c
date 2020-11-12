//------------------------------------------------------------------------------
// GB_bitmap_assign_C_template: iterate over a bitmap matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The #include'ing file defines a GB_CIJ_WORK macro for the body of the loop,
// which operates on the entry C(iC,jC) at position Cx [pC] and Cb [pC].  The C
// matrix held in bitmap form.  If the mask matrix is also a bitmap matrix or
// full matrix, the GB_GET_MIJ macro can compute the effective value of the
// mask for the C(iC,jC) entry.

// C must be bitmap or full.  If M is accessed, it must also be bitmap or full.

#ifndef GB_GET_MIJ
#define GB_GET_MIJ(mij,pM) ;
#endif

{
    switch (assign_kind)
    {

        //----------------------------------------------------------------------
        // row assignment: C<M'>(iC,:), M is a column vector
        //----------------------------------------------------------------------

        case GB_ROW_ASSIGN : 
        {
            // iterate over all of C(iC,:)
            int64_t iC = I [0] ;
            int64_t jC ;
            int nthreads = GB_nthreads (cvdim, chunk, nthreads_max) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(+:cnvals)
            for (jC = 0 ; jC < cvdim ; jC++)
            { 
                int64_t pC = iC + jC * cvlen ;
                GB_GET_MIJ (mij, jC) ;          // mij = Mask (jC)
                GB_CIJ_WORK (pC) ;              // operate on C(iC,jC)
            }
        }
        break ;

        //----------------------------------------------------------------------
        // column assignment: C<M>(:,jC), M is a column vector
        //----------------------------------------------------------------------

        case GB_COL_ASSIGN : 
        {
            // iterate over all of C(:,jC)
            int64_t iC ;
            int64_t jC = J [0] ;
            int64_t pC0 = jC * cvlen ;
            int nthreads = GB_nthreads (cvlen, chunk, nthreads_max) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(+:cnvals)
            for (iC = 0 ; iC < cvlen ; iC++)
            { 
                int64_t pC = iC + pC0 ;
                GB_GET_MIJ (mij, iC) ;          // mij = Mask (iC)
                GB_CIJ_WORK (pC) ;              // operate on C(iC,jC)
            }
        }
        break ;

        //----------------------------------------------------------------------
        // GrB_assign: C<M>(I,J), M is a matrix the same size as C
        //----------------------------------------------------------------------

        case GB_ASSIGN : 
        {
            // iterate over all of C(:,:).
            int64_t pC ;
            int nthreads = GB_nthreads (cnzmax, chunk, nthreads_max) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(+:cnvals)
            for (pC = 0 ; pC < cnzmax ; pC++)
            { 
                // int64_t iC = pC % cvlen ;
                // int64_t jC = pC / cvlen ;
                GB_GET_MIJ (mij, pC) ;          // mij = Mask (pC)
                GB_CIJ_WORK (pC) ;              // operate on C(iC,jC)
            }
        }
        break ;

        //----------------------------------------------------------------------
        // GxB_subassign: C(I,J)<M>, M is a matrix the same size as C(I,J)
        //----------------------------------------------------------------------

        #ifndef GB_NO_SUBASSIGN_CASE
        case GB_SUBASSIGN : 
        {
            // iterate over all of C(I,J)
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pC,pA)                                      \
            {                                                               \
                GB_GET_MIJ (mij, pA) ;          /* mij = Mask (pA)      */  \
                GB_CIJ_WORK (pC) ;              /* operate on C(iC,jC)  */  \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
        break ;
        #endif

        default: ;
    }
}

