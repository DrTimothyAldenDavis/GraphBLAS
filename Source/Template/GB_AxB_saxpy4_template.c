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
double ttt = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // get the max # of threads and chunk size to slice C
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // get A, B, and C
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

    int64_t *GB_RESTRICT Wi = (*Wi_handle) ;

    //==========================================================================
    // phase1: compute the pattern of C
    //==========================================================================

    int64_t cnvec_nonempty = 0 ;
    bool scan_C_to_clear_Wf = true ;

    // GB_MTYPE is only used below if M is bitmap/full and not structural
    #undef GB_MTYPE
    #undef GB_CHECK_MASK
    #undef GB_M_IS_BITMAP_OR_FULL

    if (M == NULL)
    { 

        //----------------------------------------------------------------------
        // M is not present, or present but not applied
        //----------------------------------------------------------------------

        // do not check the mask
        #define GB_CHECK_MASK(i) ;

        // if (f == 0) add C(i,j) as a new entry
        #undef  GB_IS_NEW_ENTRY
        #define GB_IS_NEW_ENTRY(f) (f == 0)
        // if C(i,j) is not a new entry, it already exists and f is always 2
        #undef  GB_IS_EXISTING_ENTRY
        #define GB_IS_EXISTING_ENTRY(f) (true)

        #include "GB_AxB_saxpy4_phase1.c"

    }
    else if (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M))
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

        #define GB_M_IS_BITMAP_OR_FULL
        const size_t msize = (Mask_struct) ? 0 : M->type->size ;
        const bool M_is_full = GB_IS_FULL (M) ;

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
            // scanned.  M is not scanned to clear Wf.  If M is full, it is
            // not structural.

            // check the mask condition, and skip C(i,j) if M(i,j) is true

            if (M_is_full)
            {
                switch (msize)
                {
                    default:
                    case 1 :    // M is bool, int8_t, or uint8_t
                        #define GB_MTYPE uint8_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 2 :    // M is int16 or uint16
                        #define GB_MTYPE uint16_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 4 :    // M is int32, uint32, or float
                        #define GB_MTYPE uint32_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 8 :    // M is int64, uint64, double, or complex float
                        #define GB_MTYPE uint64_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 16 :    // M is complex double
                        #define GB_MTYPE uint64_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [2*i] != 0 || Mxj [2*i+1] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                }
            }
            else // M is bitmap
            {
                #define GB_M_IS_BITMAP
                const int8_t *GB_RESTRICT Mb = M->b ;
                ASSERT (Mb != NULL) ;
                switch (msize)
                {
                    default:
                    case 0 :    // M is structural
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 1 :    // M is bool, int8_t, or uint8_t
                        #define GB_MTYPE uint8_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] != 0 && Mxj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 2 :    // M is int16 or uint16
                        #define GB_MTYPE uint16_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] != 0 && Mxj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 4 :    // M is int32, uint32, or float
                        #define GB_MTYPE uint32_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] != 0 && Mxj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 8 :    // M is int64, uint64, double, or complex float
                        #define GB_MTYPE uint64_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] != 0 && Mxj [i] != 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 16 :    // M is complex double
                        #define GB_MTYPE uint64_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] != 0 &&     \
                               (Mxj [2*i] != 0 || Mxj [2*i+1] != 0)) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                }
                #undef GB_M_IS_BITMAP
            }


        }
        else
        {

            //------------------------------------------------------------------
            // C<M>=A*B where M is bitmap/full
            //------------------------------------------------------------------

            // M is present, and bitmap/full.  The mask M is used in-place, not
            // scattered into Wf.  To clear Wf when done, all of C must be
            // scanned.

            // check the mask condition, and skip C(i,j) if M(i,j) is false

            if (M_is_full)
            {
                switch (msize)
                {
                    default:
                    case 1 :    // M is bool, int8_t, or uint8_t
                        #define GB_MTYPE uint8_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 2 :    // M is int16 or uint16
                        #define GB_MTYPE uint16_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 4 :    // M is int32, uint32, or float
                        #define GB_MTYPE uint32_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 8 :    // M is int64, uint64, double, or complex float
                        #define GB_MTYPE uint64_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 16 :    // M is complex double
                        #define GB_MTYPE uint64_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mxj [2*i] == 0 && Mxj [2*i+1] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                }
            }
            else // M is bitmap
            {
                #define GB_M_IS_BITMAP
                const int8_t *GB_RESTRICT Mb = M->b ;
                ASSERT (Mb != NULL) ;
                switch (msize)
                {
                    default:
                    case 0 :    // M is structural
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 1 :    // M is bool, int8_t, or uint8_t
                        #define GB_MTYPE uint8_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] == 0 || Mxj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 2 :    // M is int16 or uint16
                        #define GB_MTYPE uint16_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] == 0 || Mxj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 4 :    // M is int32, uint32, or float
                        #define GB_MTYPE uint32_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] == 0 || Mxj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 8 :    // M is int64, uint64, double, or complex float
                        #define GB_MTYPE uint64_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] == 0 || Mxj [i] == 0) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                    case 16 :    // M is complex double
                        #define GB_MTYPE uint64_t
                        #define GB_CHECK_MASK(i)    \
                            if (Mbj [i] == 0 ||     \
                               (Mxj [2*i] == 0 && Mxj [2*i+1] == 0)) continue ;
                        #include "GB_AxB_saxpy4_phase1.c"
                        break ;
                }
                #undef GB_M_IS_BITMAP
            }
        }

        #undef GB_IS_EXISTING_ENTRY
        #undef GB_IS_NEW_ENTRY
        #undef GB_M_IS_BITMAP_OR_FULL
    }

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (3, ttt) ;
ttt = omp_get_wtime ( ) ;

    //==========================================================================
    // phase2: compute numeric values of C in Hx
    //==========================================================================

    // This phase is skipped for the ANY_PAIR semiring

    #if !GB_IS_ANY_PAIR_SEMIRING

    if (nthreads > 1)
    {

        //----------------------------------------------------------------------
        // parallel case (single-threaded case handled in phase1)
        //----------------------------------------------------------------------

        if (M == NULL)
        { 
            // if no mask is present, Hf [i] will always equal 2 and so
            // it does not need to be read in.  The case for the generic
            // semiring still needs to use Hf [i] as a critical section.
            #define GB_NO_MASK
            #include "GB_AxB_saxpy4_phase2.c"
        }
        else
        { 
            // The mask is present, and accounted for in the Wf workspace
            #include "GB_AxB_saxpy4_phase2.c"
        }

    }
    #endif

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (4, ttt) ;
ttt = omp_get_wtime ( ) ;

    //==========================================================================
    // phase3: gather and sort the pattern of C
    //==========================================================================

    // Wi now contains the entire nonzero pattern of C.
    // TODO: put this in a function

    int64_t cnz ;
    if (nthreads == 1)
    {

        //----------------------------------------------------------------------
        // allocate Ci and copy Wi into Ci; Cp, cnvec_nonempty already computed
        //----------------------------------------------------------------------

        // allocate C->i
        cnz = Cp [cnvec] ;
        C->i = GB_MALLOC (GB_IMAX (cnz, 1), int64_t, &(C->i_size)) ;
        Ci = C->i ;
        if (Ci != NULL)
        { 
            memcpy (Ci, Wi, cnz * sizeof (int64_t)) ;
        }

    }
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
        GB_cumsum (Cp, cnvec, &cnvec_nonempty, nthreads, Context) ;
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

    }

    // free Wi
    GB_FREE_WERK_UNLIMITED_FROM_MALLOC (Wi_handle, Wi_size) ;

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

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (5, ttt) ;
ttt = omp_get_wtime ( ) ;

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
        #define GB_CLEAR_HF
        #include "GB_AxB_saxpy4_phase4.c"

    }
    else
    { 

        //----------------------------------------------------------------------
        // just gather C, no need to clear Wf
        //----------------------------------------------------------------------

        // skip this for the ANY_PAIR semiring
        #if !GB_IS_ANY_PAIR_SEMIRING
        #include "GB_AxB_saxpy4_phase4.c"
        #endif
    }

    // free workspace for slicing C
    GB_WERK_POP (C_ek_slicing, int64_t) ;

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (6, ttt) ;
ttt = omp_get_wtime ( ) ;
}

