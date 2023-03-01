//------------------------------------------------------------------------------
// GB_emult_02_phase1: C = A.*B where A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Symbolic analysis phsae for GB_emult_02 and GB_emult_03.

#include "GB_ewise.h"
#include "GB_emult.h"
#include "GB_binop.h"
#include "GB_stringify.h"

void GB_emult_02_phase1     // symbolic analysis for GB_emult_02 and GB_emult_03
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads,
    // workspace:
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    // output:
    int64_t *Cp_kfirst,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    const int8_t  *restrict Mb = (M == NULL) ? NULL : M->b ;
    const GB_M_TYPE *restrict Mx = (M == NULL || Mask_struct) ? NULL :
        (const GB_M_TYPE *) M->x ;
    const size_t msize = (M == NULL) ? 0 : M->type->size ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int64_t vlen = A->vlen ;
    const int64_t nvec = A->nvec ;

    const int8_t *restrict Bb = B->b ;

    int64_t *restrict Cp = C->p ;

    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // count entries in C
    //--------------------------------------------------------------------------

    // This phase is very similar to GB_select_phase1 (GB_ENTRY_SELECTOR).

    if (M == NULL)
    {

        //----------------------------------------------------------------------
        // Method2/3(a): C = A.*B where A is sparse/hyper and B is bitmap
        //----------------------------------------------------------------------

        ASSERT (GB_IS_BITMAP (B)) ;

        int tid ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < A_ntasks ; tid++)
        {
            int64_t kfirst = kfirst_Aslice [tid] ;
            int64_t klast  = klast_Aslice  [tid] ;
            Wfirst [tid] = 0 ;
            Wlast  [tid] = 0 ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                // count the entries in C(:,j)
                int64_t j = GBH (Ah, k) ;
                int64_t pB_start = j * vlen ;
                int64_t pA, pA_end ;
                GB_get_pA (&pA, &pA_end, tid, k,
                    kfirst, klast, pstart_Aslice, Ap, vlen) ;
                int64_t cjnz = 0 ;
                for ( ; pA < pA_end ; pA++)
                { 
                    cjnz += Bb [pB_start + Ai [pA]] ;
                }
                if (k == kfirst)
                { 
                    Wfirst [tid] = cjnz ;
                }
                else if (k == klast)
                { 
                    Wlast [tid] = cjnz ;
                }
                else
                { 
                    Cp [k] = cjnz ; 
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // Method2/3(c): C<#M> = A.*B; A is sparse/hyper; M, B bitmap/full
        //----------------------------------------------------------------------

        ASSERT (M != NULL) ;

        int tid ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < A_ntasks ; tid++)
        {
            int64_t kfirst = kfirst_Aslice [tid] ;
            int64_t klast  = klast_Aslice  [tid] ;
            Wfirst [tid] = 0 ;
            Wlast  [tid] = 0 ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                // count the entries in C(:,j)
                int64_t j = GBH (Ah, k) ;
                int64_t pB_start = j * vlen ;
                int64_t pA, pA_end ;
                GB_get_pA (&pA, &pA_end, tid, k,
                    kfirst, klast, pstart_Aslice, Ap, vlen) ;
                int64_t cjnz = 0 ;
                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t i = Ai [pA] ;
                    int64_t pB = pB_start + i ;
                    bool mij = GBB (Mb, pB) && GB_MCAST (Mx, pB, msize) ;
                    mij = mij ^ Mask_comp ;
                    cjnz += (mij && GBB (Bb, pB)) ;
                }
                if (k == kfirst)
                { 
                    Wfirst [tid] = cjnz ;
                }
                else if (k == klast)
                { 
                    Wlast [tid] = cjnz ;
                }
                else
                { 
                    Cp [k] = cjnz ; 
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize Cp, cumulative sum of Cp and compute Cp_kfirst
    //--------------------------------------------------------------------------

    GB_ek_slice_merge1 (Cp, Wfirst, Wlast, A_ek_slicing, A_ntasks) ;
    GB_ek_slice_merge2 (&(C->nvec_nonempty), Cp_kfirst, Cp, nvec,
        Wfirst, Wlast, A_ek_slicing, A_ntasks, A_nthreads, Werk) ;
}

