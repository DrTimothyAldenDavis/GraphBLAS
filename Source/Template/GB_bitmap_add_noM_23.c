//------------------------------------------------------------------------------
// GB_bitmap_add_noM_23: C=A+B, C bitmap, A sparse/hyper, B bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.
// A is sparse/hyper, B is bitmap.

{

    //------------------------------------------------------------------
    // Method23: C and B are bitmap; A is sparse or hypersparse
    //------------------------------------------------------------------

    #ifdef GB_ISO_ADD
        GB_memcpy (Cb, Bb, cnz, C_nthreads) ;
    #else
        int64_t p ;
        #pragma omp parallel for num_threads(C_nthreads) \
            schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            int8_t b = Bb [p] ;
            if (b)
            {
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = alpha + B(i,j)
                    GB_LOAD_B (bij, Bx, p, B_iso) ;
                    GB_BINOP (GB_CX (p), alpha_scalar, bij,
                        p % vlen, p / vlen) ;
                }
                #else
                { 
                    // C (i,j) = B (i,j)
                    GB_COPY_B_TO_C (Cx, p, Bx, p, B_iso) ;
                }
                #endif
            }
            Cb [p] = b ;
        }
    #endif

    cnvals = B->nvals ;

    GB_SLICE_MATRIX (A, 8, chunk) ;

    #pragma omp parallel for num_threads(A_nthreads) \
        schedule(dynamic,1) reduction(+:cnvals)
    for (taskid = 0 ; taskid < A_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Aslice [taskid] ;
        int64_t klast  = klast_Aslice  [taskid] ;
        int64_t task_cnvals = 0 ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of A(:,k) for this task
            int64_t j = GBH_A (Ah, k) ;
            int64_t pA_start, pA_end ;
            GB_get_pA (&pA_start, &pA_end, taskid, k, kfirst,
                klast, pstart_Aslice, Ap, vlen) ;
            int64_t pC_start = j * vlen ;
            // traverse over A(:,j), the kth vector of A
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = Ai [pA] ;
                int64_t p = pC_start + i ;
                if (Cb [p])
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    #ifndef GB_ISO_ADD
                    GB_LOAD_A (aij, Ax, pA, A_iso) ;
                    GB_LOAD_B (bij, Bx, p , B_iso) ;
                    GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                    #endif
                }
                else
                { 
                    #ifndef GB_ISO_ADD
                    #if GB_IS_EWISEUNION
                    { 
                        // C (i,j) = A(i,j) + beta
                        GB_LOAD_A (aij, Ax, pA, A_iso) ;
                        GB_BINOP (GB_CX (p), aij, beta_scalar, i, j) ;
                    }
                    #else
                    { 
                        // C (i,j) = A (i,j)
                        GB_COPY_A_TO_C (Cx, p, Ax, pA, A_iso) ;
                    }
                    #endif
                    #endif
                    Cb [p] = 1 ;
                    task_cnvals++ ;
                }
            }
        }
        cnvals += task_cnvals ;
    }
}

