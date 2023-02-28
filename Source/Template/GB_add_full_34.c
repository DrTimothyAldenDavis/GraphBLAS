//------------------------------------------------------------------------------
// GB_add_full_34:  C=A+B; C and B are full, A is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // Method34: C and B are full; A is hypersparse or sparse
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    {
        #if GB_IS_EWISEUNION
        { 
            // C (i,j) = alpha + B(i,j)
            GB_LOAD_B (bij, Bx, p, B_iso) ;
            GB_BINOP (GB_CX (p), alpha_scalar, bij, p % vlen, p / vlen) ;
        }
        #else
        { 
            // C (i,j) = B (i,j)
            GB_COPY_B_TO_C (Cx, p, Bx, p, B_iso) ;
        }
        #endif
    }

    GB_SLICE_MATRIX (A, 8, chunk) ;

    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < A_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Aslice [taskid] ;
        int64_t klast  = klast_Aslice  [taskid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of A(:,k) for this task
            int64_t j = GBH_A (Ah, k) ;
//          int64_t pA_start, pA_end ;
//          GB_get_pA (&pA_start, &pA_end, taskid, k, kfirst,
//              klast, pstart_Aslice, Ap, vlen) ;
            GB_GET_PA (pA_start, pA_end, taskid, k, kfirst,
                klast, pstart_Aslice,
                GBP_A (Ap, k, vlen), GBP_A (Ap, k+1, vlen)) ;
            int64_t pC_start = j * vlen ;
            // traverse over A(:,j), the kth vector of A
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                // C (i,j) = A (i,j) + B (i,j)
                int64_t i = Ai [pA] ;
                int64_t p = pC_start + i ;
                GB_LOAD_A (aij, Ax, pA, A_iso) ;
                GB_LOAD_B (bij, Bx, p , B_iso) ;
                GB_BINOP (GB_CX (p), aij, bij, i, j) ;
            }
        }
    }
}

