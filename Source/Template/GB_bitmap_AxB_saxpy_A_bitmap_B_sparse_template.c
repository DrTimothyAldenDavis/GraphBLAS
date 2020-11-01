//------------------------------------------------------------------------------
// GB_bitmap_AxB_saxpy_A_bitmap_B_sparse: C<#M>+=A*B, C bitmap, M any format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task to compute C (istart:iend-1, j1:j2-1)
        //----------------------------------------------------------------------

        int a_tid = tid / nbslice ;
        int b_tid = tid % nbslice ;
        int64_t istart, iend ; 
        GB_PARTITION (istart, iend, avlen, a_tid, naslice) ;
        int64_t kfirst = B_slice [b_tid] ;          // defines j1
        int64_t klast = B_slice [b_tid + 1] ;       // defines j2

        //----------------------------------------------------------------------
        // C<#M>(istart:iend-1, j1:j2-1) += A(istart:iend-1,:) * B(:,j1:j2-1)
        //----------------------------------------------------------------------

        for (int64_t kk = kfirst ; kk < klast ; kk++)
        {

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            int64_t j = GBH (Bh, kk) ;      // j will be in the range j1:j2-1
            int64_t pB = Bp [kk] ;          // ok: B is sparse
            int64_t pB_end = Bp [kk+1] ;
            int64_t pC_start = j * avlen ;  // get pointer to C(:,j)
            GB_GET_T_FOR_SECONDJ ;          // prepare to iterate over B(:,j)

            //------------------------------------------------------------------
            // C<#M>(istart:iend-1,j) += A(istart:iend-1,:)*B(:,j)
            //------------------------------------------------------------------

            for ( ; pB < pB_end ; pB++)             // scan B(:,j)
            {
                int64_t k = Bi [pB] ;               // get B(k,j)
                GB_GET_B_kj ;                       // bkj = B(k,j)
                int64_t pA_start = avlen * k ;      // get pointer to A(:,k)

                //--------------------------------------------------------------
                // C(istart:iend-1,j) += A(istart:iend-1,k)*B(k,j)
                //--------------------------------------------------------------

                for (int64_t i = istart ; i < iend ; i++)
                {

                    //----------------------------------------------------------
                    // get A(i,k): pointer and bitmap status
                    //----------------------------------------------------------

                    int64_t pA = pA_start + i ;
                    if (!GBB (Ab, pA)) continue ;

                    //----------------------------------------------------------
                    // get C(i,j): pointer
                    //----------------------------------------------------------

                    int64_t pC = pC_start + i ;

                    //----------------------------------------------------------
                    // check M(i,j)
                    //----------------------------------------------------------

                    #if defined ( GB_MASK_IS_SPARSE )

                        // M is sparse or hypersparse
                        int8_t cb = Cb [pC] ;           // ok: C is bitmap
                        bool mij = (cb & 2) ;
                        if (Mask_comp) mij = !mij ;
                        if (!mij) continue ;
                        cb = (cb & 1) ;

                    #elif defined ( GB_MASK_IS_BITMAP )

                        // M is bitmap or full
                        GB_GET_M_ij (pC) ;
                        if (Mask_comp) mij = !mij ;
                        if (!mij) continue ;
                        int8_t cb = Cb [pC] ;           // ok: C is bitmap

                    #else

                        // no mask
                        int8_t cb = Cb [pC] ;           // ok: C is bitmap

                    #endif

                    //----------------------------------------------------------
                    // C(i,j) += A(i,k)*B(k,j)
                    //----------------------------------------------------------

                    GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
                    if (cb == 0)
                    { 
                        // C(i,j) = A(i,k) * B(k,j)
                        GB_CIJ_WRITE (pC, t) ;
                        Cb [pC] = keep ;
                        cnvals++ ;
                    }
                    else
                    { 
                        // C(i,j) += A(i,k) * B(k,j)
                        GB_CIJ_UPDATE (pC, t) ;
                    }
                }
            }
        }
    }
}

#undef GB_MASK_IS_SPARSE
#undef GB_MASK_IS_BITMAP

