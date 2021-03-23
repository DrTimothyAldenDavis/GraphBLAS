//------------------------------------------------------------------------------
// GB_AxB_saxpy4_phase1: compute the pattern of C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    #ifdef GB_MTYPE
    // M is bitmap/full, and not structural
    const GB_MTYPE *GB_RESTRICT Mx = (const GB_MTYPE *) M->x ;
    #endif

    if (nthreads == 1)
    {

        //----------------------------------------------------------------------
        // single-threaded case: symbolic and numeric
        //----------------------------------------------------------------------

        // When using a single thread, Wi is constructed in packed form, with
        // the (kk)th vector C(:,kk) as Wi [Cp [kk]...Cp[kk+1]-1], and Wi is
        // transplanted into Ci when done.  The numerical values of C are also
        // computed in Hx in this pass.

        // for each vector B(:,j)
        int64_t pC = 0 ;
        for (int64_t kk = 0 ; kk < bnvec ; kk++)
        {

            //------------------------------------------------------------------
            // compute C(:,j) where j is the (kk)th vector of C
            //------------------------------------------------------------------

            // get B(:,j)
            int64_t j = GBH (Bh, kk) ;
            int64_t pB = Bp [kk] ;
            int64_t pB_end = Bp [kk+1] ;
            GB_GET_T_FOR_SECONDJ ;
            // log the start of C(:,j)
            int64_t pC_start = pC ;
            Cp [kk] = pC_start ;

            // get M(:,j) if M is bitmap/full
            #ifdef GB_M_IS_BITMAP_OR_FULL
            #ifdef GB_MTYPE
            const GB_MTYPE *GB_RESTRICT Mxj = Mx + j * cvlen ;
            #endif
            #ifdef GB_M_IS_BITMAP
            const int8_t *GB_RESTRICT Mbj = Mb + j * cvlen ;
            #endif
            #endif

            // get H(:,j)
            int64_t pH = kk * cvlen ;
            int8_t *GB_RESTRICT Hf = Wf + pH ;
            GB_CTYPE *GB_RESTRICT Hx = (GB_CTYPE *) (Wx + pH * GB_CSIZE) ;
            #if GB_IS_PLUS_FC32_MONOID
            float  *GB_RESTRICT Hx_real = (float *) Hx ;
            float  *GB_RESTRICT Hx_imag = Hx_real + 1 ;
            #elif GB_IS_PLUS_FC64_MONOID
            double *GB_RESTRICT Hx_real = (double *) Hx ;
            double *GB_RESTRICT Hx_imag = Hx_real + 1 ;
            #endif

            //------------------------------------------------------------------
            // for each entry B(k,j)
            //------------------------------------------------------------------

            for ( ; pB < pB_end ; pB++)
            {
                // get B(k,j)
                int64_t k = Bi [pB] ;
                GB_GET_B_kj ;
                // get A(:,k)
                int64_t pA = Ap [k] ;
                int64_t pA_end = Ap [k+1] ;
                for ( ; pA < pA_end ; pA++)
                {
                    // get A(i,k)
                    int64_t i = Ai [pA] ;
                    // check M(i,j) if M is bitmap/full
                    #ifdef GB_M_IS_BITMAP_OR_FULL
                    GB_CHECK_MASK (i) ;
                    #endif
                    int8_t f = Hf [i] ;
                    if (GB_IS_NEW_ENTRY (f))
                    {
                        // C(i,j) is a new entry in C
                        // C(i,j) = A(i,k) * B(k,j)
                        GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                        GB_HX_WRITE (i, t) ;    // Hx [i] = t 
                        Wi [pC++] = i ;         // add i to pattern of C(:,j)
                        Hf [i] = 2 ;            // flag C(i,j) as seen
                    }
                    #if !GB_IS_ANY_MONOID
                    else if (GB_IS_EXISTING_ENTRY (f))
                    {
                        // C(i,j) is an existing entry in C
                        // C(i,j) += A(i,k) * B(k,j)
                        GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t 
                    }
                    #endif
                }
            }

            //------------------------------------------------------------------
            // count the number of nonempty vectors in C and sort if requested
            //------------------------------------------------------------------

            int64_t cknz = pC - pC_start ;
            if (cknz > 0)
            {
                cnvec_nonempty++ ;
                if (do_sort)
                { 
                    // sort C(:,j)
                    GB_qsort_1a (Wi + pC_start, cknz) ;
                }
            }
        }

        //----------------------------------------------------------------------
        // log the end of the last vector of C
        //----------------------------------------------------------------------

        Cp [bnvec] = pC ;

    }
    else
    {

        //----------------------------------------------------------------------
        // parallel case: symbolic only, except to clear Hx
        //----------------------------------------------------------------------

        // When using multiple threads, Wi is constructed in unpacked form,
        // with the (kk)th vector C(:,kk) as Wi [kk*cvlen ... Cp [kk]-1].
        // The numerical values of C are not computed in phase1, but in
        // phase2 instead.

        int taskid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (taskid = 0 ; taskid < nthreads ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor: vectors kfirst:klast
            //------------------------------------------------------------------

            // each task has local int64_t array Ci_local [1024], on the stack,
            // to hold part of the pattern of C(:,j) for a single thread.
            #undef  GB_CI_LOCAL_LEN
            #define GB_CI_LOCAL_LEN 1024
            int64_t Ci_local [GB_CI_LOCAL_LEN] ;
            // find the first and last vectors of this slice of B
            GB_ek_slice_search (taskid, nthreads, pstart_Bslice,
                Bp, bnvec, bvlen, kfirst_Bslice, klast_Bslice) ;

            // for each vector B(:,j) in this task
            int64_t kfirst = kfirst_Bslice [taskid] ;
            int64_t klast  = klast_Bslice  [taskid] ;
            for (int64_t kk = kfirst ; kk <= klast ; kk++)
            {

                //--------------------------------------------------------------
                // compute pattern of C(:,j) where j is the (kk)th vector of C
                //--------------------------------------------------------------

                // get B(:,j)
                int64_t j = GBH (Bh, kk) ;
                int64_t pB, pB_end ;
                GB_get_pA (&pB, &pB_end, taskid, kk,
                    kfirst, klast, pstart_Bslice, Bp, bvlen) ;

                // get M(:,j) if M is bitmap/full
                #ifdef GB_M_IS_BITMAP_OR_FULL
                #ifdef GB_MTYPE
                const GB_MTYPE *GB_RESTRICT Mxj = Mx + j * cvlen ;
                #endif
                #ifdef GB_M_IS_BITMAP
                const int8_t *GB_RESTRICT Mbj = Mb + j * cvlen ;
                #endif
                #endif

                // get H(:,j)
                int64_t pH = kk * cvlen ;
                int8_t *GB_RESTRICT Hf = Wf + pH ;
                #if !GB_IS_ANY_MONOID
                GB_CTYPE *GB_RESTRICT Hx = (GB_CTYPE *) (Wx + pH * GB_CSIZE) ;
                #endif
                #if GB_IS_PLUS_FC32_MONOID
                float  *GB_RESTRICT Hx_real = (float *) Hx ;
                float  *GB_RESTRICT Hx_imag = Hx_real + 1 ;
                #elif GB_IS_PLUS_FC64_MONOID
                double *GB_RESTRICT Hx_real = (double *) Hx ;
                double *GB_RESTRICT Hx_imag = Hx_real + 1 ;
                #endif

                // clear the contents of Ci_local
                int e = 0 ;

                //--------------------------------------------------------------
                // for each entry B(k,j)
                //--------------------------------------------------------------

                for ( ; pB < pB_end ; pB++)
                {
                    // get B(k,j)
                    int64_t k = Bi [pB] ;
                    // get A(:,k)
                    int64_t pA = Ap [k] ;
                    int64_t pA_end = Ap [k+1] ;
                    for ( ; pA < pA_end ; pA++)
                    {
                        // get A(i,k)
                        int64_t i = Ai [pA] ;
                        int8_t f ;
                        // check M(i,j)
                        GB_CHECK_MASK (i) ;
                        // capture and set Hf (i)
                        // atomic: { f = Hf [i] ; Hf [i] = 2 ; }
                        GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 2) ;
                        if (GB_IS_NEW_ENTRY (f))
                        {
                            // C(i,j) is a new entry in C
                            Ci_local [e++] = i ;
                            if (e == GB_CI_LOCAL_LEN)
                            {
                                // flush Ci_local and clear Hx
                                int64_t pC ;
                                // TODO:: use something else on Windows
                                GB_ATOMIC_CAPTURE
                                {
                                    pC = Cp [kk] ; Cp [kk] += GB_CI_LOCAL_LEN ;
                                }
                                memcpy (Wi + pC, Ci_local,
                                    GB_CI_LOCAL_LEN * sizeof (int64_t)) ;
                                #if !GB_IS_ANY_MONOID
                                GB_PRAGMA_SIMD_VECTORIZE
                                for (int s = 0 ; s < GB_CI_LOCAL_LEN ; s++)
                                {
                                    // Hx [Ci_local [s]] = identity
                                    GB_HX_CLEAR (Ci_local [s]) ;
                                }
                                #endif
                                e = 0 ;
                            }
                        }
                    }
                }

                //--------------------------------------------------------------
                // flush the contents of Ci_local [0:e-1]
                //--------------------------------------------------------------

                if (e > 0)
                {
                    // flush Ci_local and clear Hx
                    int64_t pC ;
                    GB_ATOMIC_CAPTURE
                    {
                        pC = Cp [kk] ; Cp [kk] += e ;
                    }
                    memcpy (Wi + pC, Ci_local, e * sizeof (int64_t)) ;
                    #if !GB_IS_ANY_MONOID
                    GB_PRAGMA_SIMD_VECTORIZE
                    for (int s = 0 ; s < e ; s++)
                    {
                        // Hx [Ci_local [s]] = identity
                        GB_HX_CLEAR (Ci_local [s]) ;
                    }
                    #endif
                }
            }
        }
    }
}

#undef GB_MTYPE
#undef GB_CHECK_MASK

