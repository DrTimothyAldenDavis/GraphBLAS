//------------------------------------------------------------------------------
// GB_bitmap_add_template: C = A+B, C<M>=A+B, and C<!M>=A+B, C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C is bitmap.  The mask M can have any sparsity structure, and is efficient
// to apply (all methods are asymptotically optimal).  All cases (no M, M, !M)
// are handled.

{

    int64_t p, cnvals = 0 ;

    if (M == NULL)
    {

        //----------------------------------------------------------------------
        // M is not present
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C       =           A       +       B
        //      ------------------------------------------
        //      bitmap  .           sparse          bitmap
        //      bitmap  .           bitmap          sparse
        //      bitmap  .           bitmap          bitmap

        ASSERT (A_is_bitmap || B_is_bitmap) ;
        ASSERT (!A_is_full) ;
        ASSERT (!B_is_full) ;

        if (A_is_bitmap && B_is_bitmap)
        {

            //------------------------------------------------------------------
            // Method21: C, A, and B are all bitmap
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
                reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                int8_t c = 0 ;
                if (Ab [p] && Bb [p])
                {   GB_cov[1080]++ ;
// covered (1080): 39728
                    // C (i,j) = A (i,j) + B (i,j)
                    GB_GETA (aij, Ax, p) ;
                    GB_GETB (bij, Bx, p) ;
                    GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                    c = 1 ;
                }
                else if (Bb [p])
                {   GB_cov[1081]++ ;
// covered (1081): 38490
                    // C (i,j) = B (i,j)
                    GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                    c = 1 ;
                }
                else if (Ab [p])
                {   GB_cov[1082]++ ;
// covered (1082): 41511
                    // C (i,j) = A (i,j)
                    GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                    c = 1 ;
                }
                Cb [p] = c ;
                cnvals += c ;
            }

        }
        else if (A_is_bitmap)
        {

            //------------------------------------------------------------------
            // Method22: C and A are bitmap; B is sparse or hypersparse
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(C_nthreads) schedule(static)
            for (p = 0 ; p < cnz ; p++)
            {   GB_cov[1083]++ ;
// covered (1083): 77151
                // C (i,j) = A (i,j)
                int8_t a = Ab [p] ;
                if (a) GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                Cb [p] = a ;
            }
            cnvals = A->nvals ;

            GB_SLICE_MATRIX (B, 8) ;

            #pragma omp parallel for num_threads(B_nthreads) \
                schedule(dynamic,1) reduction(+:cnvals)
            for (taskid = 0 ; taskid < B_ntasks ; taskid++)
            {
                int64_t kfirst = kfirst_Bslice [taskid] ;
                int64_t klast  = klast_Bslice  [taskid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    // find the part of B(:,k) for this task
                    int64_t j = GBH (Bh, k) ;
                    int64_t pB_start, pB_end ;
                    GB_get_pA (&pB_start, &pB_end, taskid, k, kfirst,
                        klast, pstart_Bslice, Bp, vlen) ;
                    int64_t pC_start = j * vlen ;
                    // traverse over B(:,j), the kth vector of B
                    for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                    {
                        int64_t i = Bi [pB] ;
                        int64_t p = pC_start + i ;
                        if (Cb [p])
                        {   GB_cov[1084]++ ;
// covered (1084): 5341
                            // C (i,j) = A (i,j) + B (i,j)
                            GB_GETA (aij, Ax, p) ;
                            GB_GETB (bij, Bx, pB) ;
                            GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                        }
                        else
                        {   GB_cov[1085]++ ;
// covered (1085): 19334
                            // C (i,j) = B (i,j)
                            GB_COPY_B_TO_C (GB_CX (p), Bx, pB) ;
                            Cb [p] = 1 ;
                            cnvals++ ;
                        }
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // Method23: C and B are bitmap; A is sparse or hypersparse
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(C_nthreads) schedule(static)
            for (p = 0 ; p < cnz ; p++)
            {   GB_cov[1086]++ ;
// covered (1086): 8524492
                // C (i,j) = B (i,j)
                int8_t b = Bb [p] ;
                if (b) GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                Cb [p] = b ;
            }
            cnvals = B->nvals ;

            GB_SLICE_MATRIX (A, 8) ;

            #pragma omp parallel for num_threads(C_nthreads) \
                schedule(dynamic,1) reduction(+:cnvals)
            for (taskid = 0 ; taskid < A_ntasks ; taskid++)
            {
                int64_t kfirst = kfirst_Aslice [taskid] ;
                int64_t klast  = klast_Aslice  [taskid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    // find the part of A(:,k) for this task
                    int64_t j = GBH (Ah, k) ;
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
                        {   GB_cov[1087]++ ;
// covered (1087): 6630502
                            // C (i,j) = A (i,j) + B (i,j)
                            GB_GETA (aij, Ax, pA) ;
                            GB_GETB (bij, Bx, p) ;
                            GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                        }
                        else
                        {   GB_cov[1088]++ ;
// covered (1088): 152666
                            // C (i,j) = A (i,j)
                            GB_COPY_A_TO_C (GB_CX (p), Ax, pA) ;
                            Cb [p] = 1 ;
                            cnvals++ ;
                        }
                    }
                }
            }
        }

    }
    else if (M_is_sparse_or_hyper)
    {

        //----------------------------------------------------------------------
        // C is bitmap, M is sparse or hyper and complemented
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C     <!M> =        A       +       B
        //      ------------------------------------------
        //      bitmap  sparse      sparse          bitmap
        //      bitmap  sparse      sparse          full  
        //      bitmap  sparse      bitmap          sparse
        //      bitmap  sparse      bitmap          bitmap
        //      bitmap  sparse      bitmap          full  
        //      bitmap  sparse      full            sparse
        //      bitmap  sparse      full            bitmap
        //      bitmap  sparse      full            full  

        // M is sparse and complemented.  If M is sparse and not
        // complemented, then C is constructed as sparse, not bitmap.
        ASSERT (Mask_comp) ;

        // C(i,j) = A(i,j) + B(i,j) can only be computed where M(i,j) is
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
                    {   GB_cov[1089]++ ;
// NOT COVERED (1089):
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
        // compute C<!M>=A+B using the mask scattered in C
        //----------------------------------------------------------------------

        bool M_cleared = false ;

        if ((A_is_bitmap || A_is_full) && (B_is_bitmap || B_is_full))
        {

            //------------------------------------------------------------------
            // Method24(!M,sparse): C is bitmap, both A and B are bitmap or full
            //------------------------------------------------------------------

// TODO     #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
// TODO         reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                int8_t c = Cb [p] ;
                if (c == 0)
                {
                    // M(i,j) is zero, so C(i,j) can be computed
                    int8_t a = GBB (Ab, p) ;
                    int8_t b = GBB (Bb, p) ;
                    if (a && b)
                    {   GB_cov[1090]++ ;
// NOT COVERED (1090):
GB_GOTCHA ;
                        // C (i,j) = A (i,j) + B (i,j)
                        GB_GETA (aij, Ax, p) ;
                        GB_GETB (bij, Bx, p) ;
                        GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                        c = 1 ;
                    }
                    else if (b)
                    {   GB_cov[1091]++ ;
// NOT COVERED (1091):
GB_GOTCHA ;
                        // C (i,j) = B (i,j)
                        GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                        c = 1 ;
                    }
                    else if (a)
                    {   GB_cov[1092]++ ;
// NOT COVERED (1092):
GB_GOTCHA ;
                        // C (i,j) = A (i,j)
                        GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                        c = 1 ;
                    }
                    Cb [p] = c ;
                    cnvals += c ;
                }
                else
                {   GB_cov[1093]++ ;
// NOT COVERED (1093):
GB_GOTCHA ;
                    // M(i,j) == 1, so C(i,j) is not computed
                    Cb [p] = 0 ;
                }
            }
            M_cleared = true ;      // M has also been cleared from C

        }
        else if (A_is_bitmap || A_is_full)
        {

            //------------------------------------------------------------------
            // Method25(!M,sparse): C bitmap, A bitmap or full, B sparse/hyper
            //------------------------------------------------------------------

// TODO     #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
// TODO         reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                if (Cb [p] == 0)
                {   GB_cov[1094]++ ;
// NOT COVERED (1094):
GB_GOTCHA ;
                    // C (i,j) = A (i,j)
                    int8_t a = GBB (Ab, p) ;
                    if (a) GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                    Cb [p] = a ;
                    cnvals += a ;
                }
            }

            GB_SLICE_MATRIX (B, 8) ;

            #pragma omp parallel for num_threads(B_nthreads) \
                schedule(dynamic,1) reduction(+:cnvals)
            for (taskid = 0 ; taskid < B_ntasks ; taskid++)
            {
                int64_t kfirst = kfirst_Bslice [taskid] ;
                int64_t klast  = klast_Bslice  [taskid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    // find the part of B(:,k) for this task
                    int64_t j = GBH (Bh, k) ;
                    int64_t pB_start, pB_end ;
                    GB_get_pA (&pB_start, &pB_end, taskid, k, kfirst,
                        klast, pstart_Bslice, Bp, vlen) ;
                    int64_t pC_start = j * vlen ;
                    // traverse over B(:,j), the kth vector of B
                    for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                    {
                        int64_t i = Bi [pB] ;
                        int64_t p = pC_start + i ;
                        int8_t c = Cb [p] ;
                        if (c == 1)
                        {   GB_cov[1095]++ ;
// NOT COVERED (1095):
GB_GOTCHA ;
                            // C (i,j) = A (i,j) + B (i,j)
                            GB_GETA (aij, Ax, p) ;
                            GB_GETB (bij, Bx, pB) ;
                            GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                        }
                        else if (c == 0)
                        {   GB_cov[1096]++ ;
// NOT COVERED (1096):
GB_GOTCHA ;
                            // C (i,j) = B (i,j)
                            GB_COPY_B_TO_C (GB_CX (p), Bx, pB) ;
                            Cb [p] = 1 ;
                            cnvals++ ;
                        }
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // Method26: C bitmap, A sparse or hypersparse, B bitmap or full
            //------------------------------------------------------------------

// TODO     #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
// TODO         reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                if (Cb [p] == 0)
                {   GB_cov[1097]++ ;
// NOT COVERED (1097):
GB_GOTCHA ;
                    // C (i,j) = B (i,j)
                    int8_t b = GBB (Bb, b) ;
                    if (b) GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                    Cb [p] = b ;
                    cnvals += b ;
                }
            }

            GB_SLICE_MATRIX (A, 8) ;

// TODO     #pragma omp parallel for num_threads(A_nthreads) \
// TODO         schedule(dynamic,1) reduction(+:cnvals)
            for (taskid = 0 ; taskid < A_ntasks ; taskid++)
            {
                int64_t kfirst = kfirst_Aslice [taskid] ;
                int64_t klast  = klast_Aslice  [taskid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    // find the part of A(:,k) for this task
                    int64_t j = GBH (Ah, k) ;
                    int64_t pA_start, pA_end ;
                    GB_get_pA (&pA_start, &pA_end, taskid, k, kfirst,
                        klast, pstart_Aslice, Ap, vlen) ;
                    int64_t pC_start = j * vlen ;
                    // traverse over A(:,j), the kth vector of A
                    for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                    {
                        int64_t i = Ai [pA] ;
                        int64_t p = pC_start + i ;
                        int8_t c = Cb [p] ;
                        if (c == 1)
                        {   GB_cov[1098]++ ;
// NOT COVERED (1098):
GB_GOTCHA ;
                            // C (i,j) = A (i,j) + B (i,j)
                            GB_GETA (aij, Ax, pA) ;
                            GB_GETB (bij, Bx, p) ;
                            GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                        }
                        else if (c == 0)
                        {   GB_cov[1099]++ ;
// NOT COVERED (1099):
GB_GOTCHA ;
                            // C (i,j) = A (i,j)
                            GB_COPY_A_TO_C (GB_CX (p), Ax, pA) ;
                            Cb [p] = 1 ;
                            cnvals++ ;
                        }
                    }
                }
            }
        }

        //---------------------------------------------------------------------
        // clear M from C
        //---------------------------------------------------------------------

        if (!M_cleared)
        {
            // This step is required if either A or B are sparse/hyper (if
            // one is sparse/hyper, the other must be bitmap).  It requires
            // an extra pass over the mask M, so this might be slower than
            // postponing the application of the mask, and doing it later.

// TODO     #pragma omp parallel for num_threads(M_nthreads) schedule(dynamic,1)
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
                        {   GB_cov[1100]++ ;
// NOT COVERED (1100):
GB_GOTCHA ;
                            int64_t i = Mi [pM] ;
                            int64_t p = pC_start + i ;
                            Cb [p] = 0 ;
                        }
                    }
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C is bitmap; M is bitmap or full
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C      <M> =        A       +       B
        //      ------------------------------------------
        //      bitmap  bitmap      sparse          bitmap
        //      bitmap  bitmap      sparse          full  
        //      bitmap  bitmap      bitmap          sparse
        //      bitmap  bitmap      bitmap          bitmap
        //      bitmap  bitmap      bitmap          full  
        //      bitmap  bitmap      full            sparse
        //      bitmap  bitmap      full            bitmap
        //      bitmap  bitmap      full            full  

        //      ------------------------------------------
        //      C      <M> =        A       +       B
        //      ------------------------------------------
        //      bitmap  full        sparse          bitmap
        //      bitmap  full        sparse          full  
        //      bitmap  full        bitmap          sparse
        //      bitmap  full        bitmap          bitmap
        //      bitmap  full        bitmap          full  
        //      bitmap  full        full            sparse
        //      bitmap  full        full            bitmap
        //      bitmap  full        full            full  

        //      ------------------------------------------
        //      C     <!M> =        A       +       B
        //      ------------------------------------------
        //      bitmap  bitmap      sparse          sparse
        //      bitmap  bitmap      sparse          bitmap
        //      bitmap  bitmap      sparse          full  
        //      bitmap  bitmap      bitmap          sparse
        //      bitmap  bitmap      bitmap          bitmap
        //      bitmap  bitmap      bitmap          full  
        //      bitmap  bitmap      full            sparse
        //      bitmap  bitmap      full            bitmap
        //      bitmap  bitmap      full            full  

        //      ------------------------------------------
        //      C     <!M> =        A       +       B
        //      ------------------------------------------
        //      bitmap  full        sparse          sparse
        //      bitmap  full        sparse          bitmap
        //      bitmap  full        sparse          full  
        //      bitmap  full        bitmap          sparse
        //      bitmap  full        bitmap          bitmap
        //      bitmap  full        bitmap          full  
        //      bitmap  full        full            sparse
        //      bitmap  full        full            bitmap
        //      bitmap  full        full            full  


        ASSERT (M_is_bitmap || M_is_full) ;
        ASSERT (A_is_bitmap || A_is_full || B_is_bitmap || B_is_full) ;

        #undef  GB_GET_MIJ     
        #define GB_GET_MIJ(p)                                           \
            bool mij = GBB (Mb, p) && GB_mcast (Mx, p, msize) ;         \
            if (Mask_comp) mij = !mij ;

        if ((A_is_bitmap || A_is_full) && (B_is_bitmap || B_is_full))
        {

            //------------------------------------------------------------------
            // Method27: C is bitmap; M, A, and B are bitmap or full
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
                reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                GB_GET_MIJ (p) ;
                if (mij)
                {
                    // M(i,j) is true, so C(i,j) can be computed
                    int8_t a = GBB (Ab, p) ;
                    int8_t b = GBB (Bb, p) ;
                    int8_t c = 0 ;
                    if (a && b)
                    {   GB_cov[1101]++ ;
// covered (1101): 82380
                        // C (i,j) = A (i,j) + B (i,j)
                        GB_GETA (aij, Ax, p) ;
                        GB_GETB (bij, Bx, p) ;
                        GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                        c = 1 ;
                    }
                    else if (b)
                    {   GB_cov[1102]++ ;
// covered (1102): 36574
                        // C (i,j) = B (i,j)
                        GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                        c = 1 ;
                    }
                    else if (a)
                    {   GB_cov[1103]++ ;
// covered (1103): 33620
                        // C (i,j) = A (i,j)
                        GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                        c = 1 ;
                    }
                    Cb [p] = c ;
                    cnvals += c ;
                }
                else
                {   GB_cov[1104]++ ;
// covered (1104): 117088
                    // M(i,j) == 1, so C(i,j) is not computed
                    Cb [p] = 0 ;
                }
            }

        }
        else if (A_is_bitmap || A_is_full)
        {

            //------------------------------------------------------------------
            // Method28: C bitmap; M and A bitmap or full; B sparse or hyper
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
                reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                GB_GET_MIJ (p) ;
                if (mij)
                {   GB_cov[1105]++ ;
// covered (1105): 23712
                    // C (i,j) = A (i,j)
                    int8_t a = GBB (Ab, p) ;
                    if (a) GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                    Cb [p] = a ;
                    cnvals += a ;
                }
                else
                {   GB_cov[1106]++ ;
// covered (1106): 85836
                    Cb [p] = 0 ;
                }
            }

            GB_SLICE_MATRIX (B, 8) ;

            #pragma omp parallel for num_threads(B_nthreads) \
                schedule(dynamic,1) reduction(+:cnvals)
            for (taskid = 0 ; taskid < B_ntasks ; taskid++)
            {
                int64_t kfirst = kfirst_Bslice [taskid] ;
                int64_t klast  = klast_Bslice  [taskid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    // find the part of B(:,k) for this task
                    int64_t j = GBH (Bh, k) ;
                    int64_t pB_start, pB_end ;
                    GB_get_pA (&pB_start, &pB_end, taskid, k, kfirst,
                        klast, pstart_Bslice, Bp, vlen) ;
                    int64_t pC_start = j * vlen ;
                    // traverse over B(:,j), the kth vector of B
                    for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                    {
                        int64_t i = Bi [pB] ;
                        int64_t p = pC_start + i ;
                        GB_GET_MIJ (p) ;
                        if (mij)
                        {
                            int8_t c = Cb [p] ;
                            if (c == 1)
                            {   GB_cov[1107]++ ;
// covered (1107): 2344
                                // C (i,j) = A (i,j) + B (i,j)
                                GB_GETA (aij, Ax, p) ;
                                GB_GETB (bij, Bx, pB) ;
                                GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                            }
                            else
                            {   GB_cov[1108]++ ;
// covered (1108): 8045
                                // C (i,j) = B (i,j)
                                GB_COPY_B_TO_C (GB_CX (p), Bx, pB) ;
                                Cb [p] = 1 ;
                                cnvals++ ;
                            }
                        }
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // Method29: C bitmap; M and B bitmap or full; A sparse or hyper
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
                reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                GB_GET_MIJ (p) ;
                if (mij)
                {   GB_cov[1109]++ ;
// covered (1109): 23782
                    // C (i,j) = B (i,j)
                    int8_t b = GBB (Bb, p) ;
                    if (b) GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                    Cb [p] = b ;
                    cnvals += b ;
                }
                else
                {   GB_cov[1110]++ ;
// covered (1110): 86313
                    Cb [p] = 0 ;
                }
            }

            GB_SLICE_MATRIX (A, 8) ;

            #pragma omp parallel for num_threads(A_nthreads) \
                schedule(dynamic,1) reduction(+:cnvals)
            for (taskid = 0 ; taskid < A_ntasks ; taskid++)
            {
                int64_t kfirst = kfirst_Aslice [taskid] ;
                int64_t klast  = klast_Aslice  [taskid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    // find the part of A(:,k) for this task
                    int64_t j = GBH (Ah, k) ;
                    int64_t pA_start, pA_end ;
                    GB_get_pA (&pA_start, &pA_end, taskid, k, kfirst,
                        klast, pstart_Aslice, Ap, vlen) ;
                    int64_t pC_start = j * vlen ;
                    // traverse over A(:,j), the kth vector of A
                    for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                    {
                        int64_t i = Ai [pA] ;
                        int64_t p = pC_start + i ;
                        GB_GET_MIJ (p) ;
                        if (mij)
                        {
                            int8_t c = Cb [p] ;
                            if (c == 1)
                            {   GB_cov[1111]++ ;
// covered (1111): 2351
                                // C (i,j) = A (i,j) + B (i,j)
                                GB_GETA (aij, Ax, pA) ;
                                GB_GETB (bij, Bx, p) ;
                                GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                            }
                            else
                            {   GB_cov[1112]++ ;
// covered (1112): 8427
                                // C (i,j) = A (i,j)
                                GB_COPY_A_TO_C (GB_CX (p), Ax, pA) ;
                                Cb [p] = 1 ;
                                cnvals++ ;
                            }
                        }
                    }
                }
            }
        }
    }

    C->nvals = cnvals ;
}

