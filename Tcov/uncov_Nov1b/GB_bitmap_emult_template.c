//------------------------------------------------------------------------------
// GB_bitmap_emult_template: C = A.*B, C<M>=A.*B, and C<!M>=A.*B, C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C is bitmap.  The mask M can have any sparsity structure, and is efficient
// to apply (all methods are asymptotically optimal).  A and B are bitmap or
// full (with at least one of them bitmap).  All cases (no M, M, !M) are
// handled.

{

    ASSERT (A_is_bitmap || B_is_bitmap) ;
    ASSERT (A_is_bitmap || A_is_full) ;
    ASSERT (B_is_bitmap || B_is_full) ;

    // TODO modify this method so it can modify C in-place, and also use the
    // accum operator.
    int64_t p, cnvals = 0 ;

    if (M == NULL)
    {

        //----------------------------------------------------------------------
        // M is not present
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C       =           A       .*      B
        //      ------------------------------------------
        //      bitmap  .           bitmap          bitmap
        //      bitmap  .           bitmap          full  
        //      bitmap  .           full            bitmap

        //----------------------------------------------------------------------
        // Method18: C bitmap, A and B are bitmap or full
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
            reduction(+:cnvals)
        for (p = 0 ; p < cnz ; p++)
        {
            if (GBB (Ab, p) && GBB (Bb,p))
            {   GB_cov[1132]++ ;
// covered (1132): 33580
                // C (i,j) = A (i,j) + B (i,j)
                GB_GETA (aij, Ax, p) ;
                GB_GETB (bij, Bx, p) ;
                GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                Cb [p] = 1 ;
                cnvals++ ;
            }
        }

    }
    else if (M_is_sparse_or_hyper)
    {

        //----------------------------------------------------------------------
        // C is bitmap, M is sparse or hyper
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C       <!M>=       A       .*      B
        //      ------------------------------------------
        //      bitmap  sparse      bitmap          bitmap
        //      bitmap  sparse      bitmap          full  
        //      bitmap  sparse      full            bitmap

        // M is sparse and complemented.  If M is sparse and not
        // complemented, then C is constructed as sparse, not bitmap.
        ASSERT (Mask_comp) ;

        // C(i,j) = A(i,j) .* B(i,j) can only be computed where M(i,j) is
        // not present in the sparse pattern of M, and where it is present
        // but equal to zero.

        //----------------------------------------------------------------------
        // scatter M into the C bitmap
        //----------------------------------------------------------------------

        GB_SLICE_MATRIX (M, 8) ;

// TODO #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
        for (taskid = 0 ; taskid < M_ntasks ; taskid++)
        {
            int64_t kfirst = kfirst_Mslice [taskid] ;
            int64_t klast  = klast_Mslice  [taskid] ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                // find the part of M(:,k) for this task
                int64_t j = GBH (Mh, k) ;
                int64_t pM_start, pM_end ;
                GB_get_pA (&pM_start, &pM_end, taskid, k, kfirst,
                    klast, pstart_Mslice, Mp, vlen) ;
                int64_t pC_start = j * vlen ;
                // traverse over M(:,j), the kth vector of M
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    // mark C(i,j) if M(i,j) is true
                    bool mij = GB_mcast (Mx, pM, msize) ;
                    if (mij)
                    {   GB_cov[1133]++ ;
// NOT COVERED (1133):
GB_GOTCHA ;
                        int64_t i = Mi [pM] ;
                        int64_t p = pC_start + i ;
                        Cb [p] = 2 ;
                    }
                }
            }
        }

        // C(i,j) has been marked, in Cb, with the value 2 where M(i,j)=1.
        // These positions will not be computed in C(i,j).  C(i,j) can only
        // be modified where Cb [p] is zero.

        //----------------------------------------------------------------------
        // Method19(!M,sparse): C is bitmap, both A and B are bitmap or full
        //----------------------------------------------------------------------

// TODO #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
// TODO     reduction(+:cnvals)
        for (p = 0 ; p < cnz ; p++)
        {
            if (Cb [p] == 0)
            {
                // M(i,j) is zero, so C(i,j) can be computed
                if (GBB (Ab, p) && GBB (Bb, p))
                {   GB_cov[1134]++ ;
// NOT COVERED (1134):
GB_GOTCHA ;
                    // C (i,j) = A (i,j) + B (i,j)
                    GB_GETA (aij, Ax, p) ;
                    GB_GETB (bij, Bx, p) ;
                    GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                    Cb [p] = 1 ;
                    cnvals++ ;
                }
            }
            else
            {   GB_cov[1135]++ ;
// NOT COVERED (1135):
GB_GOTCHA ;
                // M(i,j) == 1, so C(i,j) is not computed
                Cb [p] = 0 ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C is bitmap; M is bitmap or full
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C      <M> =        A       .*      B
        //      ------------------------------------------
        //      bitmap  bitmap      bitmap          bitmap
        //      bitmap  bitmap      bitmap          full  
        //      bitmap  bitmap      full            bitmap

        //      ------------------------------------------
        //      C      <M> =        A       .*      B
        //      ------------------------------------------
        //      bitmap  full        bitmap          bitmap
        //      bitmap  full        bitmap          full  
        //      bitmap  full        full            bitmap

        //      ------------------------------------------
        //      C      <!M> =       A       .*      B
        //      ------------------------------------------
        //      bitmap  bitmap      bitmap          bitmap
        //      bitmap  bitmap      bitmap          full  
        //      bitmap  bitmap      full            bitmap

        //      ------------------------------------------
        //      C      <!M> =       A       .*      B
        //      ------------------------------------------
        //      bitmap  full        bitmap          bitmap
        //      bitmap  full        bitmap          full  
        //      bitmap  full        full            bitmap

        ASSERT (M_is_bitmap || M_is_full) ;

        #undef  GB_GET_MIJ     
        #define GB_GET_MIJ(p)                                           \
            bool mij = GBB (Mb, p) && GB_mcast (Mx, p, msize) ;         \
            if (Mask_comp) mij = !mij ;

        //----------------------------------------------------------------------
        // Method20: C is bitmap; M, A, and B are bitmap or full
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
            reduction(+:cnvals)
        for (p = 0 ; p < cnz ; p++)
        {
            GB_GET_MIJ (p) ;
            if (mij)
            {
                // M(i,j) is true, so C(i,j) can be computed
                if (GBB (Ab, p) && GBB (Bb, p))
                {
                    // C (i,j) = A (i,j) + B (i,j)
                    GB_GETA (aij, Ax, p) ;
                    GB_GETB (bij, Bx, p) ;
                    GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                    Cb [p] = 1 ;
                    cnvals++ ;
                }
            }
            else
            {
                // M(i,j) == 1, so C(i,j) is not computed
                Cb [p] = 0 ;
            }
        }
    }

    C->nvals = cnvals ;
}

