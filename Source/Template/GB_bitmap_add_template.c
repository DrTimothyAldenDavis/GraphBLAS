//------------------------------------------------------------------------------
// GB_bitmap_add_template: C = A+B, C<M>=A+B, and C<!M>=A+B, C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  The mask M can have any sparsity structure, and is efficient
// to apply (all methods are asymptotically optimal).  All cases (no M, M, !M)
// are handled.

{

    // TODO: the input C can be modified in-place, if it is also bitmap
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
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    GB_GETA (aij, Ax, p) ;
                    GB_GETB (bij, Bx, p) ;
                    GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                    c = 1 ;
                }
                else if (Bb [p])
                { 
                    // C (i,j) = B (i,j)
                    GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                    c = 1 ;
                }
                else if (Ab [p])
                { 
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
            { 
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
                        { 
                            // C (i,j) = A (i,j) + B (i,j)
                            GB_GETA (aij, Ax, p) ;
                            GB_GETB (bij, Bx, pB) ;
                            GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                        }
                        else
                        { 
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
            { 
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
                        { 
                            // C (i,j) = A (i,j) + B (i,j)
                            GB_GETA (aij, Ax, pA) ;
                            GB_GETB (bij, Bx, p) ;
                            GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                        }
                        else
                        { 
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
                int64_t pC_start = j * vlen ;
                // traverse over M(:,j), the kth vector of M
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    // mark C(i,j) if M(i,j) is true
                    bool mij = GB_mcast (Mx, pM, msize) ;
                    if (mij)
                    { 
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

            #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
                reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                int8_t c = Cb [p] ;
                if (c == 0)
                {
                    // M(i,j) is zero, so C(i,j) can be computed
                    int8_t a = GBB (Ab, p) ;
                    int8_t b = GBB (Bb, p) ;
                    if (a && b)
                    { 
                        // C (i,j) = A (i,j) + B (i,j)
                        GB_GETA (aij, Ax, p) ;
                        GB_GETB (bij, Bx, p) ;
                        GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                        c = 1 ;
                    }
                    else if (b)
                    { 
                        // C (i,j) = B (i,j)
                        GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                        c = 1 ;
                    }
                    else if (a)
                    { 
                        // C (i,j) = A (i,j)
                        GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                        c = 1 ;
                    }
                    Cb [p] = c ;
                    cnvals += c ;
                }
                else
                { 
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

            #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
                reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                if (Cb [p] == 0)
                { 
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
                        { 
                            // C (i,j) = A (i,j) + B (i,j)
                            GB_GETA (aij, Ax, p) ;
                            GB_GETB (bij, Bx, pB) ;
                            GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                        }
                        else if (c == 0)
                        { 
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

            #pragma omp parallel for num_threads(C_nthreads) schedule(static) \
                reduction(+:cnvals)
            for (p = 0 ; p < cnz ; p++)
            {
                if (Cb [p] == 0)
                { 
                    // C (i,j) = B (i,j)
                    int8_t b = GBB (Bb, p) ;
                    if (b) GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                    Cb [p] = b ;
                    cnvals += b ;
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
                        int8_t c = Cb [p] ;
                        if (c == 1)
                        { 
                            // C (i,j) = A (i,j) + B (i,j)
                            GB_GETA (aij, Ax, pA) ;
                            GB_GETB (bij, Bx, p) ;
                            GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                        }
                        else if (c == 0)
                        { 
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
                    int64_t pC_start = j * vlen ;
                    // traverse over M(:,j), the kth vector of M
                    for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                    {
                        // mark C(i,j) if M(i,j) is true
                        bool mij = GB_mcast (Mx, pM, msize) ;
                        if (mij)
                        { 
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
                    { 
                        // C (i,j) = A (i,j) + B (i,j)
                        GB_GETA (aij, Ax, p) ;
                        GB_GETB (bij, Bx, p) ;
                        GB_BINOP (GB_CX (p), aij, bij, p % vlen, p / vlen) ;
                        c = 1 ;
                    }
                    else if (b)
                    { 
                        // C (i,j) = B (i,j)
                        GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                        c = 1 ;
                    }
                    else if (a)
                    { 
                        // C (i,j) = A (i,j)
                        GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                        c = 1 ;
                    }
                    Cb [p] = c ;
                    cnvals += c ;
                }
                else
                { 
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
                { 
                    // C (i,j) = A (i,j)
                    int8_t a = GBB (Ab, p) ;
                    if (a) GB_COPY_A_TO_C (GB_CX (p), Ax, p) ;
                    Cb [p] = a ;
                    cnvals += a ;
                }
                else
                { 
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
                            { 
                                // C (i,j) = A (i,j) + B (i,j)
                                GB_GETA (aij, Ax, p) ;
                                GB_GETB (bij, Bx, pB) ;
                                GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                            }
                            else
                            { 
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
                { 
                    // C (i,j) = B (i,j)
                    int8_t b = GBB (Bb, p) ;
                    if (b) GB_COPY_B_TO_C (GB_CX (p), Bx, p) ;
                    Cb [p] = b ;
                    cnvals += b ;
                }
                else
                { 
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
                            { 
                                // C (i,j) = A (i,j) + B (i,j)
                                GB_GETA (aij, Ax, pA) ;
                                GB_GETB (bij, Bx, p) ;
                                GB_BINOP (GB_CX (p), aij, bij, i, j) ;
                            }
                            else
                            { 
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

