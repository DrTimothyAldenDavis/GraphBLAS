//------------------------------------------------------------------------------
// GB_bitmap_AxB_saxpy_A_sparse_B_bitmap: C<#M>+=A*B, C bitmap, M any format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    if (use_coarse_tasks)
    {

        //----------------------------------------------------------------------
        // C<#M> += A*B using coarse tasks
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(+:cnvals)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the vectors of B and C for this coarse task
            //------------------------------------------------------------------

            int64_t jstart, jend ;
            GB_PARTITION (jstart, jend, bvdim, tid, ntasks) ;

            //------------------------------------------------------------------
            // C<#M>(:,jstart:jend-1) += A * B(:,jstart:jend-1)
            //------------------------------------------------------------------

            for (int64_t kA = 0 ; kA < anvec ; kA++)
            {

                //--------------------------------------------------------------
                // get A(:,k)
                //--------------------------------------------------------------

                int64_t k = GBH (Ah, kA) ;
                int64_t pA_start = Ap [kA] ;
                int64_t pA_end = Ap [kA+1] ;

                //--------------------------------------------------------------
                // C<#M>(:,jstart:jend-1) += A(:,k) * B(k,jstart:jend-1)
                //--------------------------------------------------------------

                for (int64_t j = jstart ; j < jend ; j++)
                {

                    //----------------------------------------------------------
                    // get B(k,j)
                    //----------------------------------------------------------

                    int64_t pB = k + j * bvlen ;    // get pointer to B(k,j)
                    if (!GBB (Bb, pB)) continue ;
                    GB_GET_B_kj ;                   // bkj = B(k,j)
                    int64_t pC_start = j * avlen ;  // get pointer to C(:,j)
                    GB_GET_T_FOR_SECONDJ ;          // t = j or j+1 for SECONDJ

                    //----------------------------------------------------------
                    // C<#M>(:,j) += A(:,k) * B(k,j)
                    //----------------------------------------------------------

                    for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                    {

                        //------------------------------------------------------
                        // get A(i,k)
                        //------------------------------------------------------

                        int64_t i = Ai [pA] ;

                        //------------------------------------------------------
                        // get C(i,j): pointer and bitmap status
                        //------------------------------------------------------

                        int64_t pC = pC_start + i ;
                        int8_t cb = Cb [pC] ;           // ok: C is bitmap

                        //------------------------------------------------------
                        // check M(i,j)
                        //------------------------------------------------------

                        #if defined ( GB_MASK_IS_SPARSE )

                            // M is sparse or hypersparse
                            bool mij = (cb & 2) ;
                            if (Mask_comp) mij = !mij ;
                            if (!mij) continue ;
                            cb = (cb & 1) ;

                        #elif defined ( GB_MASK_IS_BITMAP )

                            // M is bitmap or full
                            GB_GET_M_ij (pC) ;
                            if (Mask_comp) mij = !mij ;
                            if (!mij) continue ;

                        #endif

                        //------------------------------------------------------
                        // C(i,j) += A(i,k) * B(k,j)
                        //------------------------------------------------------

                        GB_MULT_A_ik_B_kj ;
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
    else
    {

        //----------------------------------------------------------------------
        // C<#M> += A*B using fine tasks
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(+:cnvals)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the vector of B and C for this fine task
            //------------------------------------------------------------------

            // The fine task operates on C(:,j) and B(:,j).  Its fine task
            // id ranges from 0 to nfine_tasks_per_vector-1, and determines
            // which slice of A to operate on.

            int64_t j    = tid / nfine_tasks_per_vector ;
            int fine_tid = tid % nfine_tasks_per_vector ;
            int64_t kfirst = A_slice [fine_tid] ;
            int64_t klast = A_slice [fine_tid + 1] ;
            int64_t pB_start = j * bvlen ;      // pointer to B(:,j)
            int64_t pC_start = j * avlen ;      // pointer to C(:,j)
            GB_GET_T_FOR_SECONDJ ;              // t = j or j+1 for SECONDJ*

            // for Hx Gustavason workspace: use C(:,j) in-place:
            GB_CTYPE *GB_RESTRICT Hx = ((GB_void *) Cx) + (pC_start * GB_CSIZE);
            #if GB_IS_PLUS_FC32_MONOID
            float  *GB_RESTRICT Hx_real = (float *) Hx ;
            float  *GB_RESTRICT Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID
            double *GB_RESTRICT Hx_real = (double *) Hx ;
            double *GB_RESTRICT Hx_imag = Hx_real + 1 ;
            #endif

            //------------------------------------------------------------------
            // C<#M>(:,j) += A(:,k1:kw) * B(k1:k2-1,j)
            //------------------------------------------------------------------

            for (int64_t kk = kfirst ; kk < klast ; kk++)
            {

                //--------------------------------------------------------------
                // C<#M>(:,j) += A(:,k) * B(k,j)
                //--------------------------------------------------------------

                int64_t k = GBH (Ah, kk) ;      // k in range k1:k2-1
                int64_t pB = pB_start + k ;     // get pointer to B(k,j)
                if (!GBB (Bb, pB)) continue ;   
                int64_t pA = Ap [kk] ;          // ok: A is sparse
                int64_t pA_end = Ap [kk+1] ;
                GB_GET_B_kj ;                   // bkj = B(k,j)

                for ( ; pA < pA_end ; pA++)
                {

                    //----------------------------------------------------------
                    // get A(i,k) and C(i,j)
                    //----------------------------------------------------------

                    int64_t i = Ai [pA] ;       // get A(i,k) index
                    int64_t pC = pC_start + i ; // get C(i,j) pointer
                    int8_t cb ;

                    //----------------------------------------------------------
                    // C<#M>(i,j) += A(i,k) * B(k,j)
                    //----------------------------------------------------------

                    #if defined ( GB_MASK_IS_SPARSE )

                        //------------------------------------------------------
                        // M is sparse, and scattered into the C bitmap
                        //------------------------------------------------------

                        // finite-state machine in Cb [pC]:
                        // 0:   cij not present, mij zero
                        // 1:   cij present, mij zero (keep==1 for !M)
                        // 2:   cij not present, mij one
                        // 3:   cij present, mij one (keep==3 for M)
                        // 7:   cij is locked

                        #if GB_HAS_ATOMIC

                            // if C(i,j) is already present and can be modified
                            // (cb==keep), and the monoid can be done
                            // atomically, then do the atomic update.  No need
                            // to modify Cb [pC].
                            GB_ATOMIC_READ
                            cb = Cb [pC] ;          // grab the entry
                            if (cb == keep)
                            {
                                #if !GB_IS_ANY_MONOID
                                GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                                GB_ATOMIC_UPDATE_HX (i, t) ;    // C(i,j) += t
                                #endif
                                continue ;          // C(i,j) has been updated
                            }

                        #endif

                        do  // lock the entry
                        {
                            // do this atomically:
                            // { cb = Cb [pC] ;  Cb [pC] = 7 ; }
                            GB_ATOMIC_CAPTURE_INT8 (cb, Cb [pC], 7) ;
                        } while (cb == 7) ; // lock owner gets 0, 1, 2, or 3
                        if (cb == keep-1)
                        {
                            // C(i,j) is a new entry
                            GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
                            GB_ATOMIC_WRITE_HX (i, t) ;     // C(i,j) = t
                            cnvals++ ;
                            cb = keep ;                     // keep the entry
                        }
                        else if (cb == keep)
                        {
                            // C(i,j) is already present
                            #if !GB_IS_ANY_MONOID
                            GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
                            GB_ATOMIC_UPDATE_HX (i, t) ;    // C(i,j) += t
                            #endif
                        }
                        GB_ATOMIC_WRITE
                        Cb [pC] = cb ;                  // unlock the entry

                    #else

                        //------------------------------------------------------
                        // M is not present, or present as bitmap/full form
                        //------------------------------------------------------

                        // finite-state machine in Cb [pC]:
                        // 0:   cij not present; can be written
                        // 1:   cij present; can be updated
                        // 7:   cij is locked

                        #if defined ( GB_MASK_IS_BITMAP )

                            //--------------------------------------------------
                            // M is bitmap or full, and not in the C bitmap
                            //--------------------------------------------------

                            // do not modify C(i,j) if not permitted by the mask
                            GB_GET_M_ij (pC) ;
                            if (Mask_comp) mij = !mij ;
                            if (!mij) continue ;

                        #endif

                        //------------------------------------------------------
                        // C(i,j) += A(i,j) * B(k,j)
                        //------------------------------------------------------

                        #if GB_HAS_ATOMIC

                            // if C(i,j) is already present (cb==1), and the
                            // monoid can be done atomically, then do the
                            // atomic update.  No need to modify Cb [pC].
                            GB_ATOMIC_READ
                            cb = Cb [pC] ;          // grab the entry
                            if (cb == 1)
                            {
                                #if !GB_IS_ANY_MONOID
                                GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                                GB_ATOMIC_UPDATE_HX (i, t) ;    // C(i,j) += t
                                #endif
                                continue ;          // C(i,j) has been updated
                            }

                        #endif

                        do  // lock the entry
                        {
                            // do this atomically:
                            // { cb = Cb [pC] ;  Cb [pC] = 7 ; }
                            GB_ATOMIC_CAPTURE_INT8 (cb, Cb [pC], 7) ;
                        } while (cb == 7) ; // lock owner gets 0 or 1
                        if (cb == 0)
                        {
                            // C(i,j) is a new entry
                            GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
                            GB_ATOMIC_WRITE_HX (i, t) ;     // C(i,j) = t
                            cnvals++ ;
                        }
                        else // cb == 1
                        {
                            // C(i,j) is already present
                            #if !GB_IS_ANY_MONOID
                            GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
                            GB_ATOMIC_UPDATE_HX (i, t) ;    // C(i,j) += t
                            #endif
                        }
                        GB_ATOMIC_WRITE
                        Cb [pC] = 1 ;               // unlock the entry

                    #endif

                }
            }
        }
    }
}

#undef GB_MASK_IS_SPARSE
#undef GB_MASK_IS_BITMAP

