//------------------------------------------------------------------------------
// GB_AxB_saxpy4_phase4: gather C and clear Wf for saxpy4 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    int taskid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (taskid = 0 ; taskid < C_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Cslice [taskid] ;
        int64_t klast  = klast_Cslice  [taskid] ;
        for (int64_t kk = kfirst ; kk <= klast ; kk++)
        {
            int64_t pC_start, pC_end ;
            GB_get_pA (&pC_start, &pC_end, taskid, kk,
                kfirst, klast, pstart_Cslice, Cp, cvlen) ;

            // get H(:,j)
            int64_t pH = kk * cvlen ;
            #ifdef GB_CLEAR_HF
            int8_t *GB_RESTRICT Hf = Wf + pH ;
            #endif
            #if !GB_IS_ANY_PAIR_SEMIRING
            GB_CTYPE *GB_RESTRICT Hx = (GB_CTYPE *) (Wx + pH * GB_CSIZE) ;
            #endif
            #if GB_IS_PLUS_FC32_MONOID
            float  *GB_RESTRICT Hx_real = (float *) Hx ;
            float  *GB_RESTRICT Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID
            double *GB_RESTRICT Hx_real = (double *) Hx ;
            double *GB_RESTRICT Hx_imag = Hx_real + 1 ;
            #endif

            // clear H(:,j) and gather C(:,j)
            GB_PRAGMA_SIMD_VECTORIZE
            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            {
                int64_t i = Ci [pC] ;
                #ifdef GB_CLEAR_HF
                Hf [i] = 0 ;
                #endif
                // Cx [pC] = Hx [i] ;
                GB_CIJ_GATHER (pC, i) ;
            }
        }
    }
}

#undef GB_CLEAR_HF

