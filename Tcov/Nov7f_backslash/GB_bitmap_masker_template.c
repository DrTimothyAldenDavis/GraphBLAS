//------------------------------------------------------------------------------
// GB_bitmap_masker_template:  phase2 for R = masker (C, M, Z), R is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Computes C<M>=Z or C<!M>=Z, returning the result in R, which is bitmap.
// The input matrix C is not modified.  Effectively, this computes R=C and then
// R<M>=Z or R<!M>=Z.  If the C_replace descriptor is enabled, then C has
// already been cleared, and is an empty (but non-NULL) matrix.

// phase2: computes R in a single pass

// C is sparse or hypersparse.  Z is bitmap or full.  R is bitmap.
// M has any sparsity structure.

        //      ------------------------------------------
        //      C       <!M> =       Z              R
        //      ------------------------------------------

        //      sparse  sparse      bitmap          bitmap
        //      sparse  sparse      full            bitmap

        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

        //      ------------------------------------------
        //      C       <M> =        Z              R
        //      ------------------------------------------

        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

// FUTURE:: add special cases for C==Z, C==M, and Z==M aliases

{
    
    int64_t p, rnvals = 0 ;

    ASSERT (R_sparsity == GxB_BITMAP) ;
    ASSERT (C_is_sparse || C_is_hyper) ;
    ASSERT (Z_is_bitmap || Z_is_full) ;

    //--------------------------------------------------------------------------
    // scatter C into the R bitmap
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C input to R_bitmap_masker", GB0) ;
    GB_SLICE_MATRIX (C, 8) ;

    #pragma omp parallel for num_threads(C_nthreads) schedule(dynamic,1) \
        reduction(+:rnvals)
    for (taskid = 0 ; taskid < C_ntasks ; taskid++)
    {
        int64_t kfirst = kfirst_Cslice [taskid] ;
        int64_t klast  = klast_Cslice  [taskid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            // find the part of C(:,k) for this task
            int64_t j = GBH (Ch, k) ;
            int64_t pC_start, pC_end ;
            GB_get_pA (&pC_start, &pC_end, taskid, k, kfirst,
                klast, pstart_Cslice, Cp, vlen) ;
            int64_t pR_start = j * vlen ;
            // traverse over C(:,j), the kth vector of C
            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            {   GB_cov[1143]++ ;
// covered (1143): 60
                // R(i,j) = C(i,j)
                int64_t i = Ci [pC] ;
                int64_t pR = pR_start + i ;
                Rb [pR] = 1 ;
                rnvals++ ;
                memcpy (Rx + (pR)*rsize, Cx +(pC)*rsize, rsize) ;
            }
        }
    }

    R->nvals = rnvals ;
    ASSERT_MATRIX_OK (R, "R with C scattered", GB0) ;

    //--------------------------------------------------------------------------
    // R<M>=Z or R<!M>=Z
    //--------------------------------------------------------------------------

    if (M_is_sparse || M_is_hyper)
    {

        //----------------------------------------------------------------------
        // Method05: M is sparse or hypersparse, Z bitmap/full, R bitmap
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C       <!M> =       Z              R
        //      ------------------------------------------

        //      sparse  sparse      bitmap          bitmap
        //      sparse  sparse      full            bitmap

        ASSERT (Mask_comp) ;

        //----------------------------------------------------------------------
        // scatter M into the R bitmap
        //----------------------------------------------------------------------

        GB_SLICE_MATRIX (M, 8) ;

        #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
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
                int64_t pR_start = j * vlen ;
                // traverse over M(:,j), the kth vector of M
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    // mark R(i,j) if M(i,j) is true
                    bool mij = GB_mcast (Mx, pM, msize) ;
                    if (mij)
                    {   GB_cov[1144]++ ;
// covered (1144): 50
                        int64_t i = Mi [pM] ;
                        int64_t p = pR_start + i ;
                        Rb [p] += 2 ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // R<!M>=Z, using M scattered into R
        //----------------------------------------------------------------------

        // Rb is marked as follows:
        //  0:  R(i,j) is not present, and M(i,j) is false
        //  1:  R(i,j) is present, and M(i,j) is false
        //  2:  R(i,j) is not present, and M(i,j) is true
        //  3:  R(i,j) is present, and M(i,j) is true

        // M is complemented, but shown uncomplemented in the table below since
        // that is how it is scattered into R.

        // Rb   R(i,j)  M(i,j)  Z(i,j)      modification to R(i,j)
        // 0    -       0       zij         R(i,j) = Z(i,j), new value, rnvals++
        // 0    -       0       -           do nothing
        // 1    rij     0       zij         R(i,j) = Z(i,j), overwrite
        // 1    rij     0       -           delete R(i,j), rnvals--
        // 2    -       1       zij         do nothing, set Rb to 0
        // 2    -       1       -           do nothing, set Rb to 0
        // 3    rij     1       zij         keep R(i,j), set Rb to 1
        // 3    rij     1       -           keep R(i,j), set Rb to 1

        #pragma omp parallel for num_threads(R_nthreads) schedule(static) \
            reduction(+:rnvals)
        for (p = 0 ; p < rnz ; p++)
        {
            int8_t r = Rb [p] ;
            int8_t z = GBB (Zb, p) ;
            switch (r)
            {
                case 0  : GB_cov[1145]++ ;     // R(i,j) not present, M(i,j) false
// covered (1145): 141
                    if (z)
                    {   GB_cov[1146]++ ;
// covered (1146): 84
                        // R(i,j) = Z(i,j), insert new value
                        memcpy (Rx +(p)*rsize, Zx +(p)*rsize, rsize) ;
                        Rb [p] = 1 ;
                        rnvals++ ;
                    }
                    break ;

                case 1  : GB_cov[1147]++ ;     // R(i,j) present, M(i,j) false
// covered (1147): 50
                    if (z)
                    {   GB_cov[1148]++ ;
// covered (1148): 28
                        // R(i,j) = Z(i,j), update prior value
                        memcpy (Rx +(p)*rsize, Zx +(p)*rsize, rsize) ;
                    }
                    else
                    {   GB_cov[1149]++ ;
// covered (1149): 26
                        // delete R(i,j)
                        Rb [p] = 0 ;
                        rnvals-- ;
                    }
                    break ;

                case 2  : GB_cov[1150]++ ;     // R(i,j) not present, M(i,j) true
// covered (1150): 37
                    Rb [p] = 0 ;
                    break ;

                case 3  : GB_cov[1151]++ ;     // R(i,j) present, M(i,j) true
// covered (1151): 13
                    Rb [p] = 1 ;
                    break ;

                default: ;
            }
        }

    }
    else
    {   GB_cov[1152]++ ;
// covered (1152): 1311

        //----------------------------------------------------------------------
        // Method06: M and Z are bitmap or full, R is bitmap
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C       <!M> =       Z              R
        //      ------------------------------------------

        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

        //      ------------------------------------------
        //      C       <M> =        Z              R
        //      ------------------------------------------

        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

        // Rb   R(i,j)  M(i,j)  Z(i,j)      modification to R(i,j)

        // 0    -       0       zij         do nothing
        // 0    -       0       -           do nothing
        // 1    rij     0       zij         do nothing
        // 1    rij     0       -           do nothing

        // 0    -       1       zij         R(i,j) = Z(i,j), rnvals++
        // 0    -       1       -           do nothing
        // 1    rij     1       zij         R(i,j) = Z(i,j), no change to rnvals
        // 1    rij     1       -           delete, rnvals--

// TODO#pragma omp parallel for num_threads(R_nthreads) schedule(static) \
// TODO     reduction(+:rnvals)
        for (p = 0 ; p < rnz ; p++)
        {
            bool mij = GBB (Mb, p) && GB_mcast (Mx, p, msize) ;
            if (Mask_comp) mij = !mij ;
            if (mij)
            {
                int8_t z = GBB (Zb, p) ;
                int8_t r = Rb [p] ;
                if (r)
                {
                    if (z)
                    {   GB_cov[1153]++ ;
// NOT COVERED (1153):
GB_GOTCHA ; // masker with R bitmap, C sparse, M/Z bitmap/full
                        // R(i,j) = Z(i,j), update, no change to rnvals
                        memcpy (Rx +(p)*rsize, Zx +(p)*rsize, rsize) ;
                    }
                    else
                    {   GB_cov[1154]++ ;
// NOT COVERED (1154):
GB_GOTCHA ; // masker with R bitmap, C sparse, M/Z bitmap/full  (LARGE) ]
                        // delete R(i,j)
                        Rb [p] = 0 ;
                        rnvals-- ;
                    }
                }
                else if (z)
                {   GB_cov[1155]++ ;
// covered (1155): 5825
                    // R(i,j) = Z(i,j), new entry
                    memcpy (Rx +(p)*rsize, Zx +(p)*rsize, rsize) ;
                    Rb [p] = 1 ;
                    rnvals++ ;
                }
            }
        }
    }

    R->nvals = rnvals ;
}

