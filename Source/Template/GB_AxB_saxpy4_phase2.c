//------------------------------------------------------------------------------
// GB_AxB_saxpy4_phase2: parallel numeric phase for saxpy4 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // parallel case (single-threaded case handled in phase1)
    //--------------------------------------------------------------------------

    ASSERT (nthreads > 1) ;

    // if no mask is present, Hf [i] will always equal 2 and so
    // it does not need to be read in.  The case for the generic
    // semiring still needs to use Hf [i] as a critical section.

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (taskid = 0 ; taskid < nthreads ; taskid++)
    {
        // for each vector B(:,j) in this task
        int64_t kfirst = kfirst_Bslice [taskid] ;
        int64_t klast  = klast_Bslice  [taskid] ;
        for (int64_t kk = kfirst ; kk <= klast ; kk++)
        {

            //------------------------------------------------------------------
            // compute values of C(:,j) where j is the (kk)th vector of C
            //------------------------------------------------------------------

            // get B(:,j)
            int64_t j = GBH (Bh, kk) ;
            int64_t pB, pB_end ;
            GB_get_pA (&pB, &pB_end, taskid, kk,
                kfirst, klast, pstart_Bslice, Bp, bvlen) ;
            GB_GET_T_FOR_SECONDJ ;

            // get H(:,j)
            int64_t pH = kk * cvlen ;
            int8_t   *GB_RESTRICT Hf = Wf + pH ;
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
                    int8_t f ;

                    #if GB_IS_ANY_MONOID

                        #ifndef GB_NO_MASK
                        GB_ATOMIC_READ
                        f = Hf [i] ;
                        if (f == 2)
                        #endif
                        {
                            // Hx(i,j) = A(i,k) * B(k,j)
                            GB_MULT_A_ik_B_kj ;         // t = A(i,k)*B(k,j)
                            GB_ATOMIC_WRITE_HX (i, t) ;     // Hx [i] = t 
                        }

                    #elif GB_HAS_ATOMIC

                        #ifndef GB_NO_MASK
                        GB_ATOMIC_READ
                        f = Hf [i] ;
                        if (f == 2)
                        #endif
                        {
                            // Hx(i,j) += A(i,k) * B(k,j)
                            GB_MULT_A_ik_B_kj ;         // t = A(i,k)*B(k,j)
                            GB_ATOMIC_UPDATE_HX (i, t) ;    // Hx [i] += t 
                        }

                    #else

                        do  // lock the entry
                        {
                            // do this atomically:
                            // { f = Hf [i] ; Hf [i] = 3 ; }
                            GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;
                        } while (f == 3) ;
                        #ifndef GB_NO_MASK
                        if (f == 2)
                        #endif
                        {
                            // Hx(i,j) += A(i,k) * B(k,j)
                            GB_MULT_A_ik_B_kj ;         // t = A(i,k)*B(k,j)
                            GB_ATOMIC_UPDATE_HX (i, t) ;    // Hx [i] += t 
                        }
                        // unlock the entry
                        GB_ATOMIC_WRITE
                        Hf [i] = f ;

                    #endif
                }
            }
        }
    }
}

#undef GB_NO_MASK
