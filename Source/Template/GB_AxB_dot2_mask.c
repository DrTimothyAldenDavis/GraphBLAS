//------------------------------------------------------------------------------
// GB_AxB_dot2_mask:  C<!M>=A'*B or C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    int64_t cnvals = 0 ;
    int ntasks = naslice * nbslice ;
    #if defined ( GB_PHASE_1_OF_2 )
    // if C is bitmap, only phase2 is used
    ASSERT (!C_is_bitmap) ;
    #endif

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        int a_tid = tid / nbslice ;
        int b_tid = tid % nbslice ;

        //----------------------------------------------------------------------
        // get the counts of C for this slice of A
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_1_OF_2 )
        int64_t *GB_RESTRICT C_count = C_counts [a_tid] ;
        #else
        int64_t *GB_RESTRICT C_count_start = NULL ;
        int64_t *GB_RESTRICT C_count_end   = NULL ;
        if (!C_is_bitmap)
        { 
            C_count_start = (a_tid == 0) ?         NULL : C_counts [a_tid] ;
            C_count_end   = (a_tid == naslice-1) ? NULL : C_counts [a_tid+1] ;
        }
        #endif

        //----------------------------------------------------------------------
        // C<M>=A'*B or C<!M>=A'*B via dot products
        //----------------------------------------------------------------------

        for (int64_t kB = B_slice [b_tid] ; kB < B_slice [b_tid+1] ; kB++)
        {

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            int64_t j = GBH (Bh, kB) ;
            int64_t pB_start = GBP (Bp, kB, vlen) ;
            int64_t pB_end   = GBP (Bp, kB+1, vlen) ;

            int64_t bjnz = pB_end - pB_start ;
            // no work to do if B(:,j) is empty
            if (bjnz == 0) continue ;

            //------------------------------------------------------------------
            // phase 2 of 2: get the range of entries in C(:,j) to compute
            //------------------------------------------------------------------

            #if defined ( GB_PHASE_2_OF_2 )
            // this thread computes Ci and Cx [cnz:cnz_last]
            int64_t cnz, cnz_last ;
            if (!C_is_bitmap)
            { 
                cnz = Cp [kB] +
                    ((C_count_start == NULL) ? 0 : C_count_start [kB]) ;
                cnz_last = (C_count_end == NULL) ?
                    (Cp [kB+1] - 1) : (Cp [kB] + C_count_end [kB] - 1) ;
                if (cnz > cnz_last) continue ;
            }
            #endif

            //------------------------------------------------------------------
            // get M(:,j)
            //------------------------------------------------------------------

            // find vector j in M
            int64_t pM, pM_end ;
            int64_t mpleft = 0 ;
            GB_lookup (M_is_hyper, Mh, Mp, mvlen, &mpleft, mnvec-1, j,
                &pM, &pM_end) ;
            int64_t mjnz = pM_end - pM ;
            bool mdense = (mjnz == mvlen) ;

            //------------------------------------------------------------------
            // C(:,j)<!M(:,j)> = A'*B(:,j)
            //------------------------------------------------------------------

            // get the first and last index in B(:,j)
            int64_t ib_first = GBI (Bi, pB_start, vlen) ;
            int64_t ib_last  = GBI (Bi, pB_end-1, vlen) ;

            // for each vector A(:,i):
            for (int64_t kA = A_slice [a_tid] ; kA < A_slice [a_tid+1] ; kA++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                int64_t i = GBH (Ah, kA) ;
                int64_t pA     = GBP (Ap, kA, vlen) ;
                int64_t pA_end = GBP (Ap, kA+1, vlen) ;

                //--------------------------------------------------------------
                // initialize C(i,j) if C is bitmap
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_2_OF_2 )
                if (C_is_bitmap)
                { 
                    // cnz = position of C(i,j) in the bitmap
                    cnz = i + j * cvlen ;
                    Cb [cnz] = 0 ;
                }
                #endif

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                bool mij = false ;
                if (mdense)
                { 
                    // M(:,j) is dense: M is full, bitmap, or M sparse/hyper,
                    // with a fully-populated vector M(:,j)
                    mij = GBB (Mb, pM + i) && GB_mcast (Mx, pM + i, msize) ;
                }
                else
                {
                    // M(:,j) is sparse
                    bool found ;
                    int64_t pright = pM_end - 1 ;
                    GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;
                    if (found)
                    { 
                        mij = GB_mcast (Mx, pM, msize) ;
                    }
                }

                if (Mask_comp) mij = !mij ;

                if (mij)
                { 

                    //----------------------------------------------------------
                    // C(i,j) = A(:,i)'*B(:,j)
                    //----------------------------------------------------------

                    #include "GB_AxB_dot_cij.c"
                }
            }
        }
    }

    #if defined ( GB_PHASE_2_OF_2)
    C->nvals = cnvals ;
    #endif
}

