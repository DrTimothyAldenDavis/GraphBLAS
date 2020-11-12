//------------------------------------------------------------------------------
// GB_bitmap_AxB_saxpy_A_bitmap_B_bitmap: C<#M>+=A*B, C bitmap, M any format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    int64_t tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        
        //----------------------------------------------------------------------
        // get the task to compute C (I,J)
        //----------------------------------------------------------------------

        int64_t I_tid = tid / nI_tasks ;
        int64_t J_tid = tid % nI_tasks ;

        // I = istart:iend-1
        int64_t istart = I_tid * GB_TILE_SIZE ;
        int64_t iend   = GB_IMIN (avlen, istart + GB_TILE_SIZE) ;

        // J = jstart:jend-1
        int64_t jstart = J_tid * GB_TILE_SIZE ;
        int64_t jend   = GB_IMIN (bvdim, jstart + GB_TILE_SIZE) ;

        //----------------------------------------------------------------------
        // check if any entry in the M(I,J) mask permits any change to C(I,J)
        //----------------------------------------------------------------------

        #if defined ( GB_MASK_IS_SPARSE ) || defined ( GB_MASK_IS_BITMAP )

            bool any_update_allowed = false ;

            for (int64_t j = jstart ; j < jend && !any_update_allowed ; j++)
            {
                for (int64_t i = istart ; i < iend && !any_update_allowed ; i++)
                { 

                    //----------------------------------------------------------
                    // get pointer to C(i,j) and M(i,j)
                    //----------------------------------------------------------

                    int64_t pC = j * avlen + i ;

                    //----------------------------------------------------------
                    // check M(i,j)
                    //----------------------------------------------------------

                    #if defined ( GB_MASK_IS_SPARSE )

                        // M is sparse or hypersparse
                        int8_t cb = Cb [pC] ;           // ok: C is bitmap
                        bool mij = (cb & 2) ;

                    #elif defined ( GB_MASK_IS_BITMAP )

                        // M is bitmap or full
                        GB_GET_M_ij (pC) ;

                    #endif

                    if (Mask_comp) mij = !mij ;
                    if (!mij) continue ;
                    any_update_allowed = true ;
                }
            }

            if (!any_update_allowed)
            { 
                // C(I,J) cannot be modified at all; skip it
                continue ;
            }

        #endif

        //----------------------------------------------------------------------
        // declare local storage for this task
        //----------------------------------------------------------------------

//      #ifndef GB_GENERIC
//      GB_ATYPE Ax_cache [GB_TILE_SIZE * GB_KTILE_SIZE] ;
//      #endif
//      int8_t Ab_cache [GB_TILE_SIZE * GB_KTILE_SIZE] ;
        bool Ab_any_in_row [GB_TILE_SIZE] ;

        //----------------------------------------------------------------------
        // C<#M>(I,J) += A(I,:) * B(:,J)
        //----------------------------------------------------------------------

        for (int64_t kstart = 0 ; kstart < avdim ; kstart += GB_KTILE_SIZE)
        {
            // K = kstart:kend-1
            int64_t kend = GB_IMIN (avdim, kstart + GB_KTILE_SIZE) ;

            //------------------------------------------------------------------
            // TODO: load A(I,K) into local storage
            //------------------------------------------------------------------

            // For built-in semirings, load A(I,K) into local storage of size
            // GB_TILE_SIZE * GB_KTILE_SIZE and transpose it.  Load in the
            // bitmap Ab if not NULL, and Ax if not NULL.

#if 0
            for (int64_t k = kstart ; k < kend ; k++)
            {
                for (int64_t i = istart ; i < iend ; i++)
                {
                    int64_t pA = i + k * avlen ;
                    int8_t ab = GBB (Ab, pA) ;
                    i_local = i - istart ;
                    k_local = k - kstart ;
                    Ab_cache [i_local * GB_KTILE_SIZE ...
                }
            }

            #ifndef GB_GENERIC
            #define Ax Ax_cache
            #endif
#endif

            //------------------------------------------------------------------
            // Check for entries in each row of A(I,K)
            //------------------------------------------------------------------

            if (A_is_bitmap)
            {
                for (int i = 0 ; i < GB_TILE_SIZE ; i++)
                { 
                    Ab_any_in_row [i] = false ;
                }
                for (int64_t k = kstart ; k < kend ; k++)
                {
                    for (int64_t i = istart ; i < iend ; i++)
                    { 
                        int64_t pA = i + k * avlen ;    // get pointer to A(i,k)
                        int8_t  ab = Ab [pA] ;          // ok: A is bitmap
                        // Ab_cache [(i-istart) * GB_KTILE_SIZE + (k-kstart)]
                        //      = ab ;
                        Ab_any_in_row [i-istart] |= ab ;
                    }
                }
            }

            //------------------------------------------------------------------
            // C<#M>(I,J) += A(I,K) * B(K,J)
            //------------------------------------------------------------------

            for (int64_t j = jstart ; j < jend ; j++)
            {

                //--------------------------------------------------------------
                // B is bitmap or full: check if any B(K,j) entry exists
                //--------------------------------------------------------------

                if (B_is_bitmap)
                {
                    int b = 0 ;
                    for (int64_t k = kstart ; k < kend ; k++)
                    { 
                        int64_t pB = k + j * bvlen ;    // pointer to B(k,j)
                        b += Bb [pB] ;
                    }
                    if (b == 0)
                    { 
                        // no entry exists in B(K,j)
                        continue ;
                    }
                }

                //--------------------------------------------------------------
                // C<#M>(I,j) += A(I,K) * B(K,j)
                //--------------------------------------------------------------

                GB_GET_T_FOR_SECONDJ ;

                for (int64_t i = istart ; i < iend ; i++)
                {

                    //----------------------------------------------------------
                    // skip if A(i,K) has no entries
                    //----------------------------------------------------------

                    if (A_is_bitmap && !Ab_any_in_row [i - istart])
                    { 
                        continue ;
                    }

                    //----------------------------------------------------------
                    // get C(i,j)
                    //----------------------------------------------------------

                    int64_t pC = i + j * avlen ;

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
                    // C(i,j) += A(i,K) * B(K,j)
                    //----------------------------------------------------------

                    if (cb == 0)
                    {

                        //------------------------------------------------------
                        // C(i,j) does not yet exist
                        //------------------------------------------------------

                        for (int64_t k = kstart ; k < kend ; k++)
                        {
                            int64_t pA = i + k * avlen ;    // pointer to A(i,k)
                            int64_t pB = k + j * bvlen ;    // pointer to B(k,j)
                            if (!GBB (Ab, pA)) continue ;
                            if (!GBB (Bb, pB)) continue ;
                            GB_GET_B_kj ;                   // get B(k,j)
                            GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
                            if (cb == 0)
                            { 
                                // C(i,j) = A(i,k) * B(k,j)
                                GB_CIJ_WRITE (pC, t) ;
                                Cb [pC] = keep ;
                                cb = keep ;
                                cnvals++ ;
                            }
                            else
                            { 
                                // C(i,j) += A(i,k) * B(k,j)
                                GB_CIJ_UPDATE (pC, t) ;
                            }
                        }

                    }
                    else
                    {

                        //------------------------------------------------------
                        // C(i,j) already exists
                        //------------------------------------------------------

                        for (int64_t k = kstart ; k < kend ; k++)
                        { 
                            int64_t pA = i + k * avlen ;    // pointer to A(i,k)
                            int64_t pB = k + j * bvlen ;    // pointer to B(k,j)
                            if (!GBB (Ab, pA)) continue ;
                            if (!GBB (Bb, pB)) continue ;
                            GB_GET_B_kj ;                   // get B(k,j)
                            GB_MULT_A_ik_B_kj ;             // t = A(i,k)*B(k,j)
                            // C(i,j) += A(i,k) * B(k,j)
                            GB_CIJ_UPDATE (pC, t) ;
                        }
                    }
                }
            }
        }
    }
}

#undef GB_MASK_IS_SPARSE
#undef GB_MASK_IS_BITMAP

