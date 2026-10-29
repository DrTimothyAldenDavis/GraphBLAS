//------------------------------------------------------------------------------
// GB_add_bitmap_M_sparse_26: C<!M>=A+B, C bitmap; M,A sparse/hyper, B bit/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.
// M is sparse/hyper and complemented.
// B is bitmap/full, A is sparse/hyper.

{

    //--------------------------------------------------------------------------
    // Method26: C bitmap, A sparse or hypersparse, B bitmap or full
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < C_nthreads ; tid++)
    {
        int64_t pstart, pend, task_cnvals = 0 ;
        GB_PARTITION (pstart, pend, cnz, tid, C_nthreads) ;
        for (int64_t p = pstart ; p < pend ; p++)
        {
            if (Cb [p] == 0)
            { 
                int8_t b = GBb_B (Bb, p) ;
                #ifndef GB_ISO_ADD
                if (b)
                { 
                    // C (i,j) = B (i,j), or alpha + B (i,j)
                    GB_EWISE_ALPHA_OP_BIJ (Cx, p, alpha_scalar, Bx, p, B_iso,
                        p % vlen, p / vlen) ;
                }
                #endif
                Cb [p] = b ;
                task_cnvals += b ;
            }
        }
        cnvals += task_cnvals ;
    }

    const int64_t *kfirst_Aslice = A_ek_slicing ;
    const int64_t *klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *pstart_Aslice = A_ek_slicing + A_ntasks*2 ;

    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (taskid = 0 ; taskid < A_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Aslice [taskid] ;
        int64_t klast  = klast_Aslice  [taskid] ;
        int64_t task_cnvals = 0 ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of A(:,k) for this task
            int64_t j = GBh_A (Ah, k) ;
            GB_GET_PA (pA_start, pA_end, taskid, k, kfirst, klast,
                pstart_Aslice, GB_IGET (Ap, k), GB_IGET (Ap, k+1)) ;
            int64_t pC_start = j * vlen ;
            // traverse over A(:,j), the kth vector of A
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = GB_IGET (Ai, pA) ;
                int64_t p = pC_start + i ;
                int8_t c = Cb [p] ;
                if (c == 1)
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    GB_EWISE_AIJ_OP_BIJ (Cx, p, Ax, pA, A_iso, Bx, p, B_iso,
                        i, j) ;
                }
                else if (c == 0)
                { 
                    // C (i,j) = A (i,j), or A (i,j) + beta
                    GB_EWISE_AIJ_OP_BETA (Cx, p, Ax, pA, A_iso, beta_scalar,
                        i, j) ;
                    Cb [p] = 1 ;
                    task_cnvals++ ;
                }
            }
        }
        cnvals += task_cnvals ;
    }
}

