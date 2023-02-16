//------------------------------------------------------------------------------
// GB_AxB_saxpy4_A_sparse_B_bitmap_template: C<#M>+=A*B, C bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is full. A is hyper/sparse, B is bitmap/full.

// C += A*B is computed with the accumulator identical to the monoid.

// This template is used by Template/GB_AxB_saxpy4_template.  It is not used
// for the generic case, nor for the ANY_PAIR case.  It is only used for the
// pre-generated kernels, and for the JIT.

#ifndef GB_BSIZE
#define GB_BSIZE sizeof (GB_B_TYPE)
#endif

#ifndef GB_CSIZE
#define GB_CSIZE sizeof (GB_C_TYPE)
#endif

{

    if (use_coarse_tasks)
    {

        //----------------------------------------------------------------------
        // C<#M> += A*B using coarse tasks
        //----------------------------------------------------------------------

        // number of columns in the workspace for each task
        #define GB_PANEL_SIZE 4

        if (B_iso)
        { 
            // No special cases needed.  GB_GETB handles the B iso case.
        }

        //----------------------------------------------------------------------
        // allocate workspace for each task
        //----------------------------------------------------------------------

        GB_WERK_PUSH (H_slice, ntasks, int64_t) ;
        if (H_slice == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        int64_t hwork = 0 ;
        int tid ;
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t jstart, jend ;
            GB_PARTITION (jstart, jend, bvdim, tid, ntasks) ;
            int64_t jtask = jend - jstart ;
            int64_t jpanel = GB_IMIN (jtask, GB_PANEL_SIZE) ;
            H_slice [tid] = hwork ;
            // full case needs Hx workspace only if jpanel > 1
            if (jpanel > 1)
            { 
                hwork += jpanel ;
            }
        }

        //----------------------------------------------------------------------

        #if GB_IS_ANY_PAIR_SEMIRING
        int64_t cvlenx = 0 ;
        #else
        int64_t cvlenx = cvlen * GB_CSIZE ;
        #endif
        Wcx = GB_MALLOC_WORK (hwork * cvlenx, GB_void, &Wcx_size) ;
        if (Wcx == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // C<#M> += A*B
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the vectors of B and C for this coarse task
            //------------------------------------------------------------------

            int64_t jstart, jend ;
            GB_PARTITION (jstart, jend, bvdim, tid, ntasks) ;
            int64_t jtask = jend - jstart ;
            int64_t jpanel = GB_IMIN (jtask, GB_PANEL_SIZE) ;

            //------------------------------------------------------------------
            // get the workspace for this task
            //------------------------------------------------------------------

            // Hf and Hx workspace to compute the panel of C
            #if ( !GB_IS_ANY_PAIR_SEMIRING )
            GB_C_TYPE *restrict Hx = (GB_C_TYPE *)
                (Wcx + H_slice [tid] * cvlenx) ;
            #endif

            //------------------------------------------------------------------
            // C<#M>(:,jstart:jend-1) += A * B(:,jstart:jend-1) by panel
            //------------------------------------------------------------------

            for (int64_t j1 = jstart ; j1 < jend ; j1 += jpanel)
            {

                //--------------------------------------------------------------
                // get the panel of np vectors j1:j2-1
                //--------------------------------------------------------------

                int64_t j2 = GB_IMIN (jend, j1 + jpanel) ;
                int64_t np = j2 - j1 ;

                //--------------------------------------------------------------
                // G = B(:,j1:j2-1), of size bvlen-by-np, in column major order
                //--------------------------------------------------------------

                int8_t *restrict Gb = (int8_t *) (Bb + (j1 * bvlen)) ;
                #if ( !GB_IS_ANY_PAIR_SEMIRING )
                GB_B_TYPE *restrict Gx = (GB_B_TYPE *)
                     (((GB_void *) (B->x)) +
                       (B_iso ? 0 : ((j1 * bvlen) * GB_BSIZE))) ;
                #endif

                //--------------------------------------------------------------
                // clear the panel H to compute C(:,j1:j2-1)
                //--------------------------------------------------------------

                #if ( !GB_IS_ANY_PAIR_SEMIRING )
                if (np == 1)
                { 
                    // Make H an alias to C(:,j1)
                    int64_t j = j1 ;
                    int64_t pC_start = j * cvlen ;    // get pointer to C(:,j)
                    // Hx is GB_C_TYPE, not GB_void, so pointer arithmetic on
                    // it is by units of size sizeof (GB_C_TYPE), not bytes.
                    Hx = Cx + pC_start ;
                }
                else
                { 
                    // C is full: set Hx = identity
                    int64_t nc = np * cvlen ;
                    #if GB_HAS_IDENTITY_BYTE
                        memset (Hx, GB_IDENTITY_BYTE, nc * GB_CSIZE) ;
                    #else
                        for (int64_t i = 0 ; i < nc ; i++)
                        { 
                            GB_HX_WRITE (i, zidentity) ; // Hx(i) = identity
                        }
                    #endif
                }
                #endif

                #if GB_IS_PLUS_FC32_MONOID
                float  *restrict Hx_real = (float *) Hx ;
                float  *restrict Hx_imag = Hx_real + 1 ;
                #elif GB_IS_PLUS_FC64_MONOID
                double *restrict Hx_real = (double *) Hx ;
                double *restrict Hx_imag = Hx_real + 1 ;
                #endif

                //--------------------------------------------------------------
                // H += A*G for one panel
                //--------------------------------------------------------------

                #undef GB_B_kj_PRESENT
                #if GB_B_IS_BITMAP
                #define GB_B_kj_PRESENT(b) b
                #else
                #define GB_B_kj_PRESENT(b) 1
                #endif

                #undef GB_MULT_A_ik_G_kj
                #if GB_IS_PAIR_MULTIPLIER
                    // t = A(i,k) * B (k,j) is already #defined as 1
                    #define GB_MULT_A_ik_G_kj(gkj,jj)
                #else
                    // t = A(i,k) * B (k,j)
                    #define GB_MULT_A_ik_G_kj(gkj,jj)                       \
                        GB_CIJ_DECLARE (t) ;                                \
                        GB_MULT (t, aik, gkj, i, k, j1 + jj)
                #endif

                #undef GB_HX_COMPUTE

                    #define GB_HX_COMPUTE(gkj,gb,jj)                        \
                    {                                                       \
                        /* H (i,jj) += A(i,k) * B(k,j) */                   \
                        if (GB_B_kj_PRESENT (gb))                           \
                        {                                                   \
                            /* t = A(i,k) * B (k,j) */                      \
                            GB_MULT_A_ik_G_kj (gkj, jj) ;                   \
                            /* Hx(i,jj)+=t */                               \
                            GB_HX_UPDATE (pH+jj, t) ;                       \
                        }                                                   \
                    }

                // handles both C bitmap and C full, using macros defined above
                // FIXME: make this a template when this method is split.
                switch (np)
                {

                    case 4 : 

                        for (int64_t kA = 0 ; kA < anvec ; kA++)
                        {
                            // get A(:,k)
                            const int64_t k = GBH_A (Ah, kA) ;
                            // get B(k,j1:j2-1)
                            #if GB_B_IS_BITMAP
                            const int8_t gb0 = Gb [k          ] ;
                            const int8_t gb1 = Gb [k +   bvlen] ;
                            const int8_t gb2 = Gb [k + 2*bvlen] ;
                            const int8_t gb3 = Gb [k + 3*bvlen] ;
                            if (!(gb0 || gb1 || gb2 || gb3)) continue ;
                            #endif
                            GB_DECLAREB (gk0) ;
                            GB_DECLAREB (gk1) ;
                            GB_DECLAREB (gk2) ;
                            GB_DECLAREB (gk3) ;
                            GB_GETB (gk0, Gx, k          , B_iso) ;
                            GB_GETB (gk1, Gx, k +   bvlen, B_iso) ;
                            GB_GETB (gk2, Gx, k + 2*bvlen, B_iso) ;
                            GB_GETB (gk3, Gx, k + 3*bvlen, B_iso) ;
                            // H += A(:,k)*B(k,j1:j2-1)
                            const int64_t pA_end = Ap [kA+1] ;
                            for (int64_t pA = Ap [kA] ; pA < pA_end ; pA++)
                            { 
                                const int64_t i = Ai [pA] ;
                                const int64_t pH = i * 4 ;
                                GB_DECLAREA (aik) ;
                                GB_GETA (aik, Ax, pA, A_iso) ;
                                GB_HX_COMPUTE (gk0, gb0, 0) ;
                                GB_HX_COMPUTE (gk1, gb1, 1) ;
                                GB_HX_COMPUTE (gk2, gb2, 2) ;
                                GB_HX_COMPUTE (gk3, gb3, 3) ;
                            }
                        }
                        break ;

                    case 3 : 

                        for (int64_t kA = 0 ; kA < anvec ; kA++)
                        {
                            // get A(:,k)
                            const int64_t k = GBH_A (Ah, kA) ;
                            // get B(k,j1:j2-1)
                            #if GB_B_IS_BITMAP
                            const int8_t gb0 = Gb [k          ] ;
                            const int8_t gb1 = Gb [k +   bvlen] ;
                            const int8_t gb2 = Gb [k + 2*bvlen] ;
                            if (!(gb0 || gb1 || gb2)) continue ;
                            #endif
                            GB_DECLAREB (gk0) ;
                            GB_DECLAREB (gk1) ;
                            GB_DECLAREB (gk2) ;
                            GB_GETB (gk0, Gx, k          , B_iso) ;
                            GB_GETB (gk1, Gx, k +   bvlen, B_iso) ;
                            GB_GETB (gk2, Gx, k + 2*bvlen, B_iso) ;
                            // H += A(:,k)*B(k,j1:j2-1)
                            const int64_t pA_end = Ap [kA+1] ;
                            for (int64_t pA = Ap [kA] ; pA < pA_end ; pA++)
                            { 
                                const int64_t i = Ai [pA] ;
                                const int64_t pH = i * 3 ;
                                GB_DECLAREA (aik) ;
                                GB_GETA (aik, Ax, pA, A_iso) ;
                                GB_HX_COMPUTE (gk0, gb0, 0) ;
                                GB_HX_COMPUTE (gk1, gb1, 1) ;
                                GB_HX_COMPUTE (gk2, gb2, 2) ;
                            }
                        }
                        break ;

                    case 2 : 

                        for (int64_t kA = 0 ; kA < anvec ; kA++)
                        {
                            // get A(:,k)
                            const int64_t k = GBH_A (Ah, kA) ;
                            // get B(k,j1:j2-1)
                            #if GB_B_IS_BITMAP
                            const int8_t gb0 = Gb [k          ] ;
                            const int8_t gb1 = Gb [k +   bvlen] ;
                            if (!(gb0 || gb1)) continue ;
                            #endif
                            // H += A(:,k)*B(k,j1:j2-1)
                            GB_DECLAREB (gk0) ;
                            GB_DECLAREB (gk1) ;
                            GB_GETB (gk0, Gx, k          , B_iso) ;
                            GB_GETB (gk1, Gx, k +   bvlen, B_iso) ;
                            const int64_t pA_end = Ap [kA+1] ;
                            for (int64_t pA = Ap [kA] ; pA < pA_end ; pA++)
                            { 
                                const int64_t i = Ai [pA] ;
                                const int64_t pH = i * 2 ;
                                GB_DECLAREA (aik) ;
                                GB_GETA (aik, Ax, pA, A_iso) ;
                                GB_HX_COMPUTE (gk0, gb0, 0) ;
                                GB_HX_COMPUTE (gk1, gb1, 1) ;
                            }
                        }
                        break ;

                    case 1 : 

                        for (int64_t kA = 0 ; kA < anvec ; kA++)
                        {
                            // get A(:,k)
                            const int64_t k = GBH_A (Ah, kA) ;
                            // get B(k,j1:j2-1) where j1 == j2-1
                            #if GB_B_IS_BITMAP
                            const int8_t gb0 = Gb [k] ;
                            if (!gb0) continue ;
                            #endif
                            // H += A(:,k)*B(k,j1:j2-1)
                            GB_DECLAREB (gk0) ;
                            GB_GETB (gk0, Gx, k, B_iso) ;
                            const int64_t pA_end = Ap [kA+1] ;
                            for (int64_t pA = Ap [kA] ; pA < pA_end ; pA++)
                            { 
                                const int64_t i = Ai [pA] ;
                                const int64_t pH = i ;
                                GB_DECLAREA (aik) ;
                                GB_GETA (aik, Ax, pA, A_iso) ;
                                GB_HX_COMPUTE (gk0, 1, 0) ;
                            }
                        }
                        break ;

                    default:;
                }

                #undef GB_HX_COMPUTE
                #undef GB_B_kj_PRESENT
                #undef GB_MULT_A_ik_G_kj

                //--------------------------------------------------------------
                // C<#M>(:,j1:j2-1) = H
                //--------------------------------------------------------------

                if (np == 1)
                { 
                    // Hx is already aliased to Cx; no more work to do
                    continue ;
                }

                for (int64_t jj = 0 ; jj < np ; jj++)
                {

                    //----------------------------------------------------------
                    // C<#M>(:,j) = H (:,jj)
                    //----------------------------------------------------------

                    int64_t j = j1 + jj ;
                    int64_t pC_start = j * cvlen ;  // get pointer to C(:,j)

                    for (int64_t i = 0 ; i < cvlen ; i++)
                    {
                        int64_t pC = pC_start + i ;     // pointer to C(i,j)
                        int64_t pH = i * np + jj ;      // pointer to H(i,jj)

                        //------------------------------------------------------
                        // check M(i,j)
                        //------------------------------------------------------

                        #if GB_MASK_IS_SPARSE_OR_HYPER

                            // M is sparse or hypersparse
                            bool mij = ((cb & 2) != 0) ^ Mask_comp ;
                            if (!mij) continue ;
                            cb = (cb & 1) ;

                        #elif GB_MASK_IS_BITMAP_OR_FULL

                            // M is bitmap or full
                            GB_GET_M_ij (pC) ;
                            mij = mij ^ Mask_comp ;
                            if (!mij) continue ;

                        #endif

                        //------------------------------------------------------
                        // C(i,j) += H(i,jj)
                        //------------------------------------------------------

                        { 
                            // C(i,j) = H(i,jj)
                            GB_CIJ_GATHER_UPDATE (pC, pH) ;
                        }
                    }
                }
            }
        }

        #undef GB_PANEL_SIZE

    }
    else if (use_atomics)
    {

        //----------------------------------------------------------------------
        // C<#M> += A*B using fine tasks and atomics
        //----------------------------------------------------------------------

        if (B_iso)
        { 
            // No special cases needed.  GB_GET_B_kj (bkj = B(k,j))
            // handles the B iso case.
        }

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
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
            int64_t pC_start = j * cvlen ;      // pointer to C(:,j)
            GB_GET_T_FOR_SECONDJ ;              // t = j or j+1 for SECONDJ*

            // for Hx Gustavason workspace: use C(:,j) in-place:
            #if ( !GB_IS_ANY_PAIR_SEMIRING )
            GB_C_TYPE *restrict Hx = (GB_C_TYPE *)
                (((GB_void *) Cx) + (pC_start * GB_CSIZE)) ;
            #endif
            #if GB_IS_PLUS_FC32_MONOID || GB_IS_ANY_FC32_MONOID
            float  *restrict Hx_real = (float *) Hx ;
            float  *restrict Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID || GB_IS_ANY_FC64_MONOID
            double *restrict Hx_real = (double *) Hx ;
            double *restrict Hx_imag = Hx_real + 1 ;
            #endif

            //------------------------------------------------------------------
            // C<#M>(:,j) += A(:,k1:k2) * B(k1:k2,j)
            //------------------------------------------------------------------

            for (int64_t kk = kfirst ; kk < klast ; kk++)
            {

                //--------------------------------------------------------------
                // C<#M>(:,j) += A(:,k) * B(k,j)
                //--------------------------------------------------------------

                int64_t k = GBH_A (Ah, kk) ;      // k in range k1:k2
                int64_t pB = pB_start + k ;     // get pointer to B(k,j)
                #if GB_B_IS_BITMAP
                if (!GBB_B (Bb, pB)) continue ;   
                #endif
                int64_t pA = Ap [kk] ;
                int64_t pA_end = Ap [kk+1] ;
                GB_GET_B_kj ;                   // bkj = B(k,j)

                for ( ; pA < pA_end ; pA++)
                {

                    //----------------------------------------------------------
                    // get A(i,k) and C(i,j)
                    //----------------------------------------------------------

                    int64_t i = Ai [pA] ;       // get A(i,k) index
                    int64_t pC = pC_start + i ; // get C(i,j) pointer

                    //----------------------------------------------------------
                    // C<#M>(i,j) += A(i,k) * B(k,j)
                    //----------------------------------------------------------

                    { 

                        //------------------------------------------------------
                        // C is full: the monoid is always atomic
                        //------------------------------------------------------

                        GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                        GB_Z_ATOMIC_UPDATE_HX (i, t) ;    // C(i,j) += t

                    }
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C<#M> += A*B using fine tasks and workspace, with no atomics
        //----------------------------------------------------------------------

        // Each fine task is given size-cvlen workspace to compute its result
        // in the first phase, W(:,tid) = A(:,k1:k2) * B(k1:k2,j), where k1:k2
        // is defined by the fine_tid of the task.  The workspaces are then
        // summed into C in the second phase.

        if (B_iso)
        { 
            // No special cases needed.  GB_GET_B_kj (bkj = B(k,j))
            // handles the B iso case.
        }

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        size_t workspace = cvlen * ntasks ;
        #if ( GB_IS_ANY_PAIR_SEMIRING )
        size_t cxsize = 0 ;
        #else
        size_t cxsize = GB_CSIZE ;
        #endif
        Wcx = GB_MALLOC_WORK (workspace * cxsize, GB_void, &Wcx_size) ;
        if (Wcx == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // first phase: W (:,tid) = A (:,k1:k2) * B (k2:k2,j) for each fine task
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
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
            int64_t pC_start = j * cvlen ;      // pointer to C(:,j), for bitmap
            int64_t pW_start = tid * cvlen ;    // pointer to W(:,tid)
            GB_GET_T_FOR_SECONDJ ;              // t = j or j+1 for SECONDJ*

            #if ( !GB_IS_ANY_PAIR_SEMIRING )
            GB_C_TYPE *restrict Hx = (GB_C_TYPE *) (Wcx + (pW_start * cxsize)) ;
            #endif
            #if GB_IS_PLUS_FC32_MONOID
            float  *restrict Hx_real = (float *) Hx ;
            float  *restrict Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID
            double *restrict Hx_real = (double *) Hx ;
            double *restrict Hx_imag = Hx_real + 1 ;
            #endif

            //------------------------------------------------------------------
            // clear the panel
            //------------------------------------------------------------------

            { 
                // C is full: set Hx = identity
                #if GB_HAS_IDENTITY_BYTE
                    memset (Hx, GB_IDENTITY_BYTE, cvlen * GB_CSIZE) ;
                #else
                    for (int64_t i = 0 ; i < cvlen ; i++)
                    { 
                        GB_HX_WRITE (i, zidentity) ; // Hx(i) = identity
                    }
                #endif
            }

            //------------------------------------------------------------------
            // W<#M> = A(:,k1:k2) * B(k1:k2,j)
            //------------------------------------------------------------------

            for (int64_t kk = kfirst ; kk < klast ; kk++)
            {

                //--------------------------------------------------------------
                // W<#M>(:,tid) += A(:,k) * B(k,j)
                //--------------------------------------------------------------

                int64_t k = GBH_A (Ah, kk) ;      // k in range k1:k2
                int64_t pB = pB_start + k ;     // get pointer to B(k,j)
                #if GB_B_IS_BITMAP
                if (!GBB_B (Bb, pB)) continue ;   
                #endif
                int64_t pA = Ap [kk] ;
                int64_t pA_end = Ap [kk+1] ;
                GB_GET_B_kj ;                   // bkj = B(k,j)

                for ( ; pA < pA_end ; pA++)
                {

                    //----------------------------------------------------------
                    // get A(i,k)
                    //----------------------------------------------------------

                    int64_t i = Ai [pA] ;       // get A(i,k) index

                    //----------------------------------------------------------
                    // check M(i,j)
                    //----------------------------------------------------------

                    #if GB_MASK_IS_SPARSE_OR_HYPER
                    { 
                        // M is sparse or hypersparse
                        int64_t pC = pC_start + i ;
                        int8_t cb = Cb [pC] ;
                        bool mij = ((cb & 2) != 0) ^ Mask_comp ;
                        if (!mij) continue ;
                    }
                    #elif GB_MASK_IS_BITMAP_OR_FULL
                    { 
                        // M is bitmap or full
                        int64_t pC = pC_start + i ;
                        GB_GET_M_ij (pC) ;
                        mij = mij ^ Mask_comp ;
                        if (!mij) continue ;
                    }
                    #endif

                    //----------------------------------------------------------
                    // W<#M>(i) += A(i,k) * B(k,j)
                    //----------------------------------------------------------

                    #if GB_IS_ANY_PAIR_SEMIRING
                    { 
                        Hf [i] = 1 ;
                    }
                    #else
                    {
                        GB_MULT_A_ik_B_kj ;         // t = A(i,k)*B(k,j)
                        { 
                            // W(i) is already present
                            GB_HX_UPDATE (i, t) ;   // Hx(i) += t
                        }
                    }
                    #endif
                }
            }
        }

        //----------------------------------------------------------------------
        // second phase: C<#M> += reduce (W)
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // determine the W and C for this fine task
            //------------------------------------------------------------------

            // The fine task operates on C(i1:i2,j) and W(i1:i2,w1:w2), where
            // i1:i2 is defined by the fine task id.  Its fine task id ranges
            // from 0 to nfine_tasks_per_vector-1.
            
            // w1:w2 are the updates to C(:,j), where w1:w2 =
            // [j*nfine_tasks_per_vector : (j+1)*nfine_tasks_per_vector-1].

            int64_t j    = tid / nfine_tasks_per_vector ;
            int fine_tid = tid % nfine_tasks_per_vector ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, cvlen, fine_tid,
                nfine_tasks_per_vector) ;
            int64_t pC_start = j * cvlen ;          // pointer to C(:,j)
            int64_t wstart = j * nfine_tasks_per_vector ;
            int64_t wend = (j + 1) * nfine_tasks_per_vector ;

            // Hx = (typecasted) Wcx workspace, use Wf as-is
            #if ( !GB_IS_ANY_PAIR_SEMIRING )
            GB_C_TYPE *restrict Hx = ((GB_C_TYPE *) Wcx) ;
            #endif
            #if GB_IS_PLUS_FC32_MONOID
            float  *restrict Hx_real = (float *) Hx ;
            float  *restrict Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID
            double *restrict Hx_real = (double *) Hx ;
            double *restrict Hx_imag = Hx_real + 1 ;
            #endif

            //------------------------------------------------------------------
            // C<#M>(i1:i2,j) += reduce (W (i2:i2, wstart:wend))
            //------------------------------------------------------------------

            for (int64_t w = wstart ; w < wend ; w++)
            {

                //--------------------------------------------------------------
                // C<#M>(i1:i2,j) += W (i1:i2,w)
                //--------------------------------------------------------------
            
                int64_t pW_start = w * cvlen ;      // pointer to W (:,w)

                for (int64_t i = istart ; i < iend ; i++)
                {

                    //----------------------------------------------------------
                    // get pointer and bitmap C(i,j) and W(i,w)
                    //----------------------------------------------------------

                    int64_t pW = pW_start + i ;     // pointer to W(i,w)
                    int64_t pC = pC_start + i ;     // pointer to C(i,j)

                    //----------------------------------------------------------
                    // M(i,j) already checked, but adjust Cb if M is sparse
                    //----------------------------------------------------------

                    #if GB_MASK_IS_SPARSE_OR_HYPER
                    { 
                        // M is sparse or hypersparse
                        cb = (cb & 1) ;
                    }
                    #endif

                    //----------------------------------------------------------
                    // C(i,j) += W (i,w)
                    //----------------------------------------------------------

                    { 
                        // C(i,j) += W(i,w)
                        GB_CIJ_GATHER_UPDATE (pC, pW) ;
                    }
                }
            }
        }
    }
}

