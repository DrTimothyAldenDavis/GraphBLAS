
{

    //------------------------------------------------------------------
    // Method32: C and A are full; B is sparse or hypersparse
    //------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    {
        #if GB_IS_EWISEUNION
        { 
            // C (i,j) = A(i,j) + beta
            GB_LOAD_A (aij, Ax, p, A_iso) ;
            GB_BINOP (GB_CX (p), aij, beta_scalar,
                p % vlen, p / vlen) ;
        }
        #else
        { 
            // C (i,j) = A (i,j)
            GB_COPY_A_TO_C (Cx, p, Ax, p, A_iso) ;
        }
        #endif
    }

    GB_SLICE_MATRIX (B, 8, chunk) ;

    #pragma omp parallel for num_threads(B_nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < B_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Bslice [taskid] ;
        int64_t klast  = klast_Bslice  [taskid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of B(:,k) for this task
            int64_t j = GBH_B (Bh, k) ;
            int64_t pB_start, pB_end ;
            GB_get_pA (&pB_start, &pB_end, taskid, k, kfirst,
                klast, pstart_Bslice, Bp, vlen) ;
            int64_t pC_start = j * vlen ;
            // traverse over B(:,j), the kth vector of B
            for (int64_t pB = pB_start ; pB < pB_end ; pB++)
            { 
                // C (i,j) = A (i,j) + B (i,j)
                int64_t i = Bi [pB] ;
                int64_t p = pC_start + i ;
                GB_LOAD_A (aij, Ax, p , A_iso) ;
                GB_LOAD_B (bij, Bx, pB, B_iso) ;
                GB_BINOP (GB_CX (p), aij, bij, i, j) ;
            }
        }
    }
}
