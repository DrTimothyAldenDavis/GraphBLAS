//------------------------------------------------------------------------------
// GB_AxB_saxpy4_template: compute C=A*B, C<M>=A*B, or C<!M>=A*B in parallel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// if out-of-memory, workspace and contents of C are freed in the caller
#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

{

    //--------------------------------------------------------------------------
    // get the chunk size
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    int64_t  *GB_RESTRICT Cp = C->p ;
    int64_t  *GB_RESTRICT Ci = NULL ;
    GB_CTYPE *GB_RESTRICT Cx = NULL ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnvec = C->nvec ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) (B_is_pattern ? NULL : B->x) ;
    const int64_t bvlen = B->vlen ;
    const int64_t bnvec = B->nvec ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) (A_is_pattern ? NULL : A->x) ;

    // if M is sparse/hyper, it is already scattered into Wf
    const int8_t  *GB_RESTRICT Mb = NULL ;
    const GB_void *GB_RESTRICT Mx = NULL ;
    size_t msize = 0 ;
    int64_t mnvec = 0 ;
    int64_t mvlen = 0 ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const bool M_is_sparse_or_hyper = (M != NULL && (M_is_sparse||M_is_hyper)) ;
    if (M != NULL)
    { 
        Mb = M->b ;
        Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
        mnvec = M->nvec ;
        mvlen = M->vlen ;
    }

    int64_t *GB_RESTRICT Wi = (*Wi_handle) ;

    //==========================================================================
    // phase1: compute the pattern of C
    //==========================================================================

    int64_t cnvec_nonempty = 0 ;
    bool scan_C_to_clear_Wf = true ;

    if (M == NULL)
    {

        //----------------------------------------------------------------------
        // M is not present
        //----------------------------------------------------------------------

        // do not check the mask
        #undef  GB_GET_MASK_j
        #define GB_GET_MASK_j ;
        #undef  GB_CHECK_MASK
        #define GB_CHECK_MASK(i) ;
        #undef  GB_CHECK_BITMAP_OR_FULL_MASK
        #define GB_CHECK_BITMAP_OR_FULL_MASK(i) ;
        // if (f == 0) add C(i,j) as a new entry
        #undef  GB_IS_NEW_ENTRY
        #define GB_IS_NEW_ENTRY(f) (f == 0)
        // if C(i,j) is not a new entry, it already exists and f is always 2
        #undef  GB_IS_EXISTING_ENTRY
        #define GB_IS_EXISTING_ENTRY(f) (true)
        #include "GB_AxB_saxpy4_phase1.c"

    }
    else if (M_is_sparse || M_is_hyper)
    {

        //----------------------------------------------------------------------
        // M is sparse/hyper and has been scattered into Wf
        //----------------------------------------------------------------------

        // if (f == 2) then C(i,j) already exists
        #undef  GB_IS_EXISTING_ENTRY
        #define GB_IS_EXISTING_ENTRY(f) (f == 2)

        if (Mask_comp)
        {

            //------------------------------------------------------------------
            // C<!M>=A*B
            //------------------------------------------------------------------

            // The mask is sparse and complemented.  M has been scattered into
            // Wf, with Wf [p] = M(i,j) = 0 or 1.  C(i,j) can be added to the
            // pattern if M(i,j) is zero.  To clear Wf when done, all of C and
            // M (if present) must be scanned.

            // skip this entry if M(i,j) == 1 or C(i,j) already in the pattern
            #undef  GB_CHECK_MASK
            #define GB_CHECK_MASK(i)        \
                GB_ATOMIC_READ              \
                f = Hf [i] ;                \
                if (f != 0) continue ;
            // if (f == 0) add C(i,j) as a new entry
            #undef  GB_IS_NEW_ENTRY
            #define GB_IS_NEW_ENTRY(f) (f == 0)
            #include "GB_AxB_saxpy4_phase1.c"

        }
        else
        {

            //------------------------------------------------------------------
            // C<M> = A*B
            //------------------------------------------------------------------

            // The mask M is sparse/hyper and not complemented.  M has been
            // scattered into Wf, with Wf [p] = M(i,j) = 0 or 1.  C(i,j) can be
            // added to the pattern if M(i,j) is 1.  To clear Wf when done,
            // only M needs to be scanned since the pattern of C is a subset of
            // M.  The scan of C can be skipped when clearing Wf.

            scan_C_to_clear_Wf = false ;

            // skip this entry if M(i,j) == 0 or C(i,j) already in the pattern
            #undef  GB_CHECK_MASK
            #define GB_CHECK_MASK(i)        \
                GB_ATOMIC_READ              \
                f = Hf [i] ;                \
                if (f != 1) continue ;
            // if (f == 1) add C(i,j) as a new entry
            #undef  GB_IS_NEW_ENTRY
            #define GB_IS_NEW_ENTRY(f) (f == 1)
            #include "GB_AxB_saxpy4_phase1.c"

        }
    }
    else
    {

        //----------------------------------------------------------------------
        // M is bitmap/full, and used in-place
        //----------------------------------------------------------------------

        // get M(:,j)
        #undef  GB_GET_MASK_j
        #define GB_GET_MASK_j           \
            int64_t pM = j * mvlen ;
        // if (f == 0) add C(i,j) as a new entry
        #undef  GB_IS_NEW_ENTRY
        #define GB_IS_NEW_ENTRY(f) (f == 0)
        // if C(i,j) is not a new entry, it already exists and f is always 2
        #undef  GB_IS_EXISTING_ENTRY
        #define GB_IS_EXISTING_ENTRY(f) (true)

        if (Mask_comp)
        {

            //------------------------------------------------------------------
            // C<!M>=A*B where M is bitmap/full
            //------------------------------------------------------------------

            // !M is present, and bitmap/full.  The mask M is used in-place,
            // not scattered into Wf.  To clear Wf when done, all of C must be
            // scanned.  M is not scanned to clear Wf.

            // TODO: could specialize this, for each type of mask

            // check the mask condition, and skip C(i,j) if M(i,j) is true
            #undef  GB_CHECK_MASK
            #define GB_CHECK_MASK(i)                                        \
                bool mij = GBB (Mb, pM+i) && GB_mcast (Mx, pM+i, msize) ;   \
                if (mij) continue ;
            #undef  GB_CHECK_BITMAP_OR_FULL_MASK
            #define GB_CHECK_BITMAP_OR_FULL_MASK(i) GB_CHECK_MASK(i)
            #include "GB_AxB_saxpy4_phase1.c"

        }
        else
        {

            //------------------------------------------------------------------
            // C<M>=A*B where M is bitmap/full
            //------------------------------------------------------------------

            // M is present, and bitmap/full.  The mask M is used in-place, not
            // scattered into Wf.  To clear Wf when done, all of C must be
            // scanned.

            // TODO: could specialize this, for each type of mask

            // check the mask condition, and skip C(i,j) if M(i,j) is false
            #undef  GB_CHECK_MASK
            #define GB_CHECK_MASK(i)                                        \
                bool mij = GBB (Mb, pM+i) && GB_mcast (Mx, pM+i, msize) ;   \
                if (!mij) continue ;
            #undef  GB_CHECK_BITMAP_OR_FULL_MASK
            #define GB_CHECK_BITMAP_OR_FULL_MASK(i) GB_CHECK_MASK(i)
            #include "GB_AxB_saxpy4_phase1.c"
        }
    }

    //==========================================================================
    // phase2: compute numeric values of C in Hx
    //==========================================================================

    #if !GB_IS_ANY_PAIR_SEMIRING

    if (nthreads > 1)
    {

        //----------------------------------------------------------------------
        // parallel case (single-threaded case handled in phase1)
        //----------------------------------------------------------------------

        // TODO: if no mask is present, Hf [i] will always equal 2 and so
        // it does not need to be read in.  The case for the generic
        // semiring would still need to use Hf [i] as a critical section.

        int taskid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (taskid = 0 ; taskid < nthreads ; taskid++)
        {
            // for each vector B(:,j) in this task
            int64_t kfirst = kfirst_Bslice [taskid] ;
            int64_t klast  = klast_Bslice  [taskid] ;
            for (int64_t kk = kfirst ; kk <= klast ; kk++)
            {

                //--------------------------------------------------------------
                // compute values of C(:,j) where j is the (kk)th vector of C
                //--------------------------------------------------------------

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

                //--------------------------------------------------------------
                // for each entry B(k,j)
                //--------------------------------------------------------------

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

                            GB_ATOMIC_READ
                            f = Hf [i] ;
                            if (f == 2)
                            {
                                // Hx(i,j) = A(i,k) * B(k,j)
                                GB_MULT_A_ik_B_kj ;         // t = A(i,k)*B(k,j)
                                GB_ATOMIC_WRITE_HX (i, t) ;     // Hx [i] = t 
                            }

                        #elif GB_HAS_ATOMIC

                            GB_ATOMIC_READ
                            f = Hf [i] ;
                            if (f == 2)
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
                            if (f == 2)
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

    #endif

    //==========================================================================
    // phase3: gather and sort the pattern of C
    //==========================================================================

    // Wi now contains the entire nonzero pattern of C.
    // TODO: put this in a function

    int64_t cnz ;
    if (nthreads == 1)
    {

        //----------------------------------------------------------------------
        // transplant Wi as C->i; Cp and cnvec_nonempty already computed
        //----------------------------------------------------------------------

        // C->i = Wi ;
        // Ci = C->i ;
        // Wi = NULL ;
        // (*Wi_handle) = NULL ;
        // C->i_size = Wi_size ;
        cnz = Cp [cnvec] ;

        // allocate C->i
        C->i = GB_MALLOC (GB_IMAX (cnz, 1), int64_t, &(C->i_size)) ;
        Ci = C->i ;
        if (Ci != NULL)
        {
            memcpy (Ci, Wi, cnz * sizeof (int64_t)) ;
        }
        GB_FREE_WERK_UNLIMITED_FROM_MALLOC (Wi_handle, Wi_size) ;

    }

#if 0
    else if (cnvec == 1)
    {

        //----------------------------------------------------------------------
        // transplant Wi as C->i, and compute Cp and cnvec_nonempty
        //----------------------------------------------------------------------

        C->i = Wi ;
        Ci = C->i ;
        Wi = NULL ;
        (*Wi_handle) = NULL ;
        C->i_size = Wi_size ;
        cnz = Cp [0] ;
        Cp [0] = 0 ;
        Cp [1] = cnz ;
        cnvec_nonempty = (cnz == 0) ? 0 : 1 ;
        if (do_sort)
        {
            GB_qsort_1a (C->i, cnz) ;
        }

    }
#endif

    else
    {

        //----------------------------------------------------------------------
        // allocate Ci and copy Wi into Ci; compute Cp and cnvec_nonempty
        //----------------------------------------------------------------------

        // compute cumulative sum of Cp
        for (int64_t kk = 0 ; kk < cnvec ; kk++)
        {
            Cp [kk] -= kk * cvlen ;
        }
        GB_cumsum (Cp, cnvec, &cnvec_nonempty, nthreads_max, Context) ;
        cnz = Cp [cnvec] ;

        // allocate C->i
        C->i = GB_MALLOC (GB_IMAX (cnz, 1), int64_t, &(C->i_size)) ;
        Ci = C->i ;

        if (Ci != NULL)
        {
            // move each vector from Wi to Ci
            int64_t kk ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (kk = 0 ; kk < cnvec ; kk++)
            {
                // copy Wi(:,j) into Ci(:,j) and sort if requested
                int64_t pC = Cp [kk] ;
                int64_t cknz = Cp [kk+1] - pC ;
                memcpy (Ci + pC, Wi + kk * cvlen, cknz * sizeof (int64_t)) ;
                if (do_sort)
                {
                    GB_qsort_1a (Ci + pC, cknz) ;
                }
            }
        }

        // free Wi, which was allocted by GB_MALLOC_WERK_UNLIMITED
        GB_FREE_WERK_UNLIMITED_FROM_MALLOC (Wi_handle, Wi_size) ;
    }

    // Wi is now freed, or transplanted into C
    ASSERT ((*Wi_handle) == NULL) ;

    // allocate C->x
    C->x = GB_MALLOC (GB_IMAX (cnz, 1) * GB_CSIZE, GB_void, &(C->x_size)) ;
    Cx = C->x ;

    if (Ci == NULL || Cx == NULL)
    { 
        // out of memory
        // workspace and contents of C will be freed by the caller
        return (GrB_OUT_OF_MEMORY) ;
    }

    C->nzmax = GB_IMAX (cnz, 1) ;
    C->nvec_nonempty = cnvec_nonempty ;

    //==========================================================================
    // phase4: gather C and clear Wf
    //==========================================================================

    // If GB_SLICE_MATRIX runs out of memory, C_ek_slicing will be NULL and
    // thus need not be freed.  Remaining workspace, and contents of C, will
    // be freed by the caller.

    // slice C for phase 4
    int C_nthreads, C_ntasks ;
    GB_WERK_DECLARE (C_ek_slicing, int64_t) ;
    GB_SLICE_MATRIX (C, 1) ;

    #if GB_IS_ANY_PAIR_SEMIRING
    {
        // ANY_PAIR semiring: result is purely symbolic
        int64_t pC ;
        #pragma omp parallel for num_threads(C_nthreads) schedule(static)
        for (pC = 0 ; pC < cnz ; pC++)
        { 
            Cx [pC] = GB_CTYPE_CAST (1, 0) ;
        }
    }
    #endif

    if (scan_C_to_clear_Wf)
    {

        //----------------------------------------------------------------------
        // gather C and clear Wf
        //----------------------------------------------------------------------

        // For the ANY_PAIR semiring, GB_CIJ_GATHER is empty, so all this phase
        // does is to clear Hf.  It does not modify Cx.

        int taskid ;
        #pragma omp parallel for num_threads(C_nthreads) schedule(static)
        for (taskid = 0 ; taskid < C_ntasks ; taskid++)
        {
            int64_t kfirst = kfirst_Cslice [taskid] ;
            int64_t klast  = klast_Cslice  [taskid] ;
            for (int64_t kk = kfirst ; kk <= klast ; kk++)
            {
                int64_t pC_start, pC_end ;
                GB_get_pA (&pC_start, &pC_end, taskid, kk,
                    kfirst, klast, pstart_Cslice, Cp, cvlen) ;

                // get H(:,j)
                int64_t pH = kk * cvlen ;
                int8_t *GB_RESTRICT Hf = Wf + pH ;
                #if !GB_IS_ANY_PAIR_SEMIRING
                GB_CTYPE *GB_RESTRICT Hx = (GB_CTYPE *) (Wx + pH * GB_CSIZE) ;
                #endif
                #if GB_IS_PLUS_FC32_MONOID
                float  *GB_RESTRICT Hx_real = (float *) Hx ;
                float  *GB_RESTRICT Hx_imag = Hx_real + 1 ;
                #elif GB_IS_PLUS_FC64_MONOID
                double *GB_RESTRICT Hx_real = (double *) Hx ;
                double *GB_RESTRICT Hx_imag = Hx_real + 1 ;
                #endif

                // clear H(:,j) and gather C(:,j)
                GB_PRAGMA_SIMD_VECTORIZE
                for (int64_t pC = pC_start ; pC < pC_end ; pC++)
                {
                    int64_t i = Ci [pC] ;
                    Hf [i] = 0 ;
                    // Cx [pC] = Hx [i] ;
                    GB_CIJ_GATHER (pC, i) ;
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // just gather C, no need to clear Wf
        //----------------------------------------------------------------------

        // skip this for the ANY_PAIR semiring

        #if !GB_IS_ANY_PAIR_SEMIRING

        int taskid ;
        #pragma omp parallel for num_threads(C_nthreads) schedule(static)
        for (taskid = 0 ; taskid < C_ntasks ; taskid++)
        {
            int64_t kfirst = kfirst_Cslice [taskid] ;
            int64_t klast  = klast_Cslice  [taskid] ;
            for (int64_t kk = kfirst ; kk <= klast ; kk++)
            {
                int64_t pC_start, pC_end ;
                GB_get_pA (&pC_start, &pC_end, taskid, kk,
                    kfirst, klast, pstart_Cslice, Cp, cvlen) ;

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

                // gather C(:,j)
                GB_PRAGMA_SIMD_VECTORIZE
                for (int64_t pC = pC_start ; pC < pC_end ; pC++)
                {
                    // gather C(i,j)
                    int64_t i = Ci [pC] ;
                    // Cx [pC] = Hx [i] ;
                    GB_CIJ_GATHER (pC, i) ;
                }
            }
        }

        #endif
    }

    // free workspace for slicing C
    GB_WERK_POP (C_ek_slicing, int64_t) ;
}

