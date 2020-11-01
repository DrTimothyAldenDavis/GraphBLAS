//------------------------------------------------------------------------------
// GB_AxB_saxpy3_template: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_AxB_saxpy3_template.c computes C=A*B for any semiring and matrix types,
// where C is sparse or hypersparse.

// TODO rename GB_AxB_saxpy_C_sparse_template.c or
// GB_sparse_AxB_saxpy_template.c

#include "GB_unused.h"

//------------------------------------------------------------------------------
// template code for C=A*B via the saxpy3 method
//------------------------------------------------------------------------------

{

// double ttt = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // get the chunk size
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Cp = C->p ;                // ok: C is sparse
    // const int64_t *GB_RESTRICT Ch = C->h ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnvec = C->nvec ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int8_t  *GB_RESTRICT Bb = B->b ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) (B_is_pattern ? NULL : B->x) ;
    const int64_t bvlen = B->vlen ;
    const bool B_jumbled = B->jumbled ;
    const bool B_is_sparse_or_hyper = GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B);

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int8_t  *GB_RESTRICT Ab = A->b ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const bool A_is_hyper = GB_IS_HYPER (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) (A_is_pattern ? NULL : A->x) ;
    const bool A_jumbled = A->jumbled ;

    const int64_t *GB_RESTRICT Mp = NULL ;
    const int64_t *GB_RESTRICT Mh = NULL ;
    const int8_t  *GB_RESTRICT Mb = NULL ;
    const int64_t *GB_RESTRICT Mi = NULL ;
    const GB_void *GB_RESTRICT Mx = NULL ;
    size_t msize = 0 ;
    int64_t mnvec = 0 ;
    int64_t mvlen = 0 ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    if (M != NULL)
    {   GB_cov[451]++ ;
// covered (451): 40488
        Mp = M->p ;
        Mh = M->h ;
        Mb = M->b ;
        Mi = M->i ;
        Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
        mnvec = M->nvec ;
        mvlen = M->vlen ;
    }

    // 3 cases:
    //      M not present and Mask_comp false: compute C=A*B
    //      M present     and Mask_comp false: compute C<M>=A*B
    //      M present     and Mask_comp true : compute C<!M>=A*B
    // If M is NULL on input, then Mask_comp is also false on input.

    const bool mask_is_M = (M != NULL && !Mask_comp) ;

    // ignore the mask if present, not complemented, dense and
    // used in place, structural, and not bitmap.  In this case,
    // all entries in M are true, so M can be ignored.
    const bool ignore_mask = mask_is_M && M_dense_in_place &&
        Mask_struct && !M_is_bitmap ;

    //==========================================================================
    // phase2: numeric work for fine tasks
    //==========================================================================

    // Coarse tasks: nothing to do in phase2.
    // Fine tasks: compute nnz (C(:,j)), and values in Hx via atomics.

    int taskid ;
//  TODO disabled:  #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kk = TaskList [taskid].vector ;
        int team_size = TaskList [taskid].team_size ;
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        int64_t pB     = TaskList [taskid].start ;
        int64_t pB_end = TaskList [taskid].end + 1 ;
        int64_t pleft = 0, pright = anvec-1 ;
        int64_t j = GBH (Bh, kk) ;

        GB_GET_T_FOR_SECONDJ ;

        #if !GB_IS_ANY_PAIR_SEMIRING
        GB_CTYPE *GB_RESTRICT Hx = (GB_CTYPE *) TaskList [taskid].Hx ;
        #endif

        #if GB_IS_PLUS_FC32_MONOID
        float  *GB_RESTRICT Hx_real = (float *) Hx ;
        float  *GB_RESTRICT Hx_imag = Hx_real + 1 ;
        #elif GB_IS_PLUS_FC64_MONOID
        double *GB_RESTRICT Hx_real = (double *) Hx ;
        double *GB_RESTRICT Hx_imag = Hx_real + 1 ;
        #endif

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // phase2: fine Gustavson task
            //------------------------------------------------------------------

            // Hf [i] == 0: unlocked, i has not been seen in C(:,j).
            //      Hx [i] is not initialized.
            //      M(i,j) is 0, or M is not present.
            //      if M: Hf [i] stays equal to 0 (or 3 if locked)
            //      if !M, or no M: C(i,j) is a new entry seen for 1st time

            // Hf [i] == 1: unlocked, i has not been seen in C(:,j).
            //      Hx [i] is not initialized.  M is present.
            //      M(i,j) is 1. (either M or !M case)
            //      if M: C(i,j) is a new entry seen for the first time.
            //      if !M: Hf [i] stays equal to 1 (or 3 if locked)

            // Hf [i] == 2: unlocked, i has been seen in C(:,j).
            //      Hx [i] is initialized.  This case is independent of M.

            // Hf [i] == 3: locked.  Hx [i] cannot be accessed.

            int8_t *GB_RESTRICT Hf = (int8_t *GB_RESTRICT) TaskList [taskid].Hf;

            if (M == NULL)
            {

                //--------------------------------------------------------------
                // phase2: fine Gustavson task, C(:,j)=A*B(:,j)
                //--------------------------------------------------------------

                // Hf [i] is initially 0.
                // 0 -> 3 : to lock, if i seen for first time
                // 2 -> 3 : to lock, if i seen already
                // 3 -> 2 : to unlock; now i has been seen

                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                {
                    if (!GBB (Bb, pB)) continue ;
                    int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
                    GB_GET_A_k ;                // get A(:,k)
                    if (aknz == 0) continue ;
                    GB_GET_B_kj ;               // bkj = B(k,j)
                    // scan A(:,k)
                    for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                    {
                        if (!GBB (Ab, pA)) continue ;
                        int64_t i = GBI (Ai, pA, avlen) ;  // get A(i,k)
                        GB_MULT_A_ik_B_kj ;      // t = A(i,k) * B(k,j)
                        int8_t f ;

                        #if GB_IS_ANY_MONOID

                            //--------------------------------------------------
                            // C(i,j) += t ; with the ANY monoid
                            //--------------------------------------------------

                            GB_ATOMIC_READ
                            f = Hf [i] ;            // grab the entry
                            if (f == 2) continue ;  // check if already updated
                            GB_ATOMIC_WRITE
                            Hf [i] = 2 ;                // flag the entry
                            GB_ATOMIC_WRITE_HX (i, t) ;    // Hx [i] = t

                        #else

                            //--------------------------------------------------
                            // C(i,j) += t ; with all other monoids
                            //--------------------------------------------------

                            #if GB_HAS_ATOMIC

                                // if C(i,j) is already present (f==2), and the
                                // monoid can be done atomically, then do the
                                // atomic update.  No need to modify Hf [i].
                                GB_ATOMIC_READ
                                f = Hf [i] ;        // grab the entry
                                if (f == 2)         // if true, update C(i,j)
                                {
                                    GB_ATOMIC_UPDATE_HX (i, t) ; // Hx [i] += t
                                    continue ;      // C(i,j) has been updated
                                }

                            #endif

                            do  // lock the entry
                            {   GB_cov[452]++ ;
// covered (452): 226987
                                // do this atomically:
                                // { f = Hf [i] ; Hf [i] = 3 ; }
                                GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;
                            } while (f == 3) ; // lock owner gets f=0 or 2
                            if (f == 0)
                            {   GB_cov[453]++ ;
// covered (453): 82329
                                // C(i,j) is a new entry
                                GB_ATOMIC_WRITE_HX (i, t) ;    // Hx [i] = t
                            }
                            else // f == 2
                            {   GB_cov[454]++ ;
// covered (454): 144658
                                // C(i,j) already appears in C(:,j)
                                GB_ATOMIC_UPDATE_HX (i, t) ;   // Hx [i] += t
                            }
                            GB_ATOMIC_WRITE
                            Hf [i] = 2 ;                // unlock the entry

                        #endif
                    }
                }

            }
            else if (mask_is_M)
            {

                //--------------------------------------------------------------
                // phase2: fine Gustavson task, C(:,j)<M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                // Hf [i] is 0 if M(i,j) not present or M(i,j)=0.
                // 0 -> 1 : has already been done in phase0 if M(i,j)=1.

                // 0 -> 0 : to ignore, if M(i,j)=0
                // 1 -> 3 : to lock, if i seen for first time
                // 2 -> 3 : to lock, if i seen already
                // 3 -> 2 : to unlock; now i has been seen

                GB_GET_M_j ;                // get M(:,j)
                GB_GET_M_j_RANGE (16) ;     // get first and last in M(:,j)
                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                {   GB_cov[455]++ ;
// covered (455): 75184
                    if (!GBB (Bb, pB)) continue ;
                    int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
                    GB_GET_A_k ;                // get A(:,k)
                    if (aknz == 0) continue ;
                    GB_GET_B_kj ;               // bkj = B(k,j)

                    #if GB_IS_ANY_MONOID

                        //------------------------------------------------------
                        // C(i,j) += A(i,k)*B(k,j) ; with the ANY monoid
                        //------------------------------------------------------

                        #define GB_IKJ                                         \
                            int8_t f ;                                         \
                            GB_ATOMIC_READ                                     \
                            f = Hf [i] ;            /* grab the entry */       \
                            if (f == 0 || f == 2) continue ;                   \
                            GB_ATOMIC_WRITE                                    \
                            Hf [i] = 2 ;            /* unlock the entry */     \
                            GB_MULT_A_ik_B_kj ;     /* t = A(i,k) * B(k,j) */  \
                            GB_ATOMIC_WRITE_HX (i, t) ;    /* Hx [i] = t */

                    #else

                        //------------------------------------------------------
                        // C(i,j) += A(i,k)*B(k,j) ; all other monoids
                        //------------------------------------------------------

                        #define GB_IKJ                                         \
                        {                                                      \
                            GB_MULT_A_ik_B_kj ;     /* t = A(i,k) * B(k,j) */  \
                            int8_t f ;                                         \
                            GB_ATOMIC_READ                                     \
                            f = Hf [i] ;            /* grab the entry */       \
                            if (GB_HAS_ATOMIC && (f == 2))                     \
                            {                                                  \
                                /* C(i,j) already seen; update it */           \
                                GB_ATOMIC_UPDATE_HX (i, t) ; /* Hx [i] += t */ \
                                continue ;       /* C(i,j) has been updated */ \
                            }                                                  \
                            if (f == 0) continue ; /* M(i,j)=0; ignore C(i,j)*/\
                            do  /* lock the entry */                           \
                            {                                                  \
                                /* do this atomically: */                      \
                                /* { f = Hf [i] ; Hf [i] = 3 ; } */            \
                                GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;        \
                            } while (f == 3) ; /* lock owner gets f=1 or 2 */  \
                            if (f == 1)                                        \
                            {                                                  \
                                /* C(i,j) is a new entry */                    \
                                GB_ATOMIC_WRITE_HX (i, t) ; /* Hx [i] = t */   \
                            }                                                  \
                            else /* f == 2 */                                  \
                            {                                                  \
                                /* C(i,j) already appears in C(:,j) */         \
                                GB_ATOMIC_UPDATE_HX (i, t) ; /* Hx [i] += t */ \
                            }                                                  \
                            GB_ATOMIC_WRITE                                    \
                            Hf [i] = 2 ;                /* unlock the entry */ \
                        }

                    #endif

                    GB_SCAN_M_j_OR_A_k ;
                    #undef GB_IKJ
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase2: fine Gustavson task, C(:,j)<!M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                // Hf [i] is 0 if M(i,j) not present or M(i,j)=0.
                // 0 -> 1 : has already been done in phase0 if M(i,j)=1

                // 1 -> 1 : to ignore, if M(i,j)=1
                // 0 -> 3 : to lock, if i seen for first time
                // 2 -> 3 : to lock, if i seen already
                // 3 -> 2 : to unlock; now i has been seen

                GB_GET_M_j ;                // get M(:,j)
                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                {
                    if (!GBB (Bb, pB)) continue ;
                    int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
                    GB_GET_A_k ;                // get A(:,k)
                    if (aknz == 0) continue ;
                    GB_GET_B_kj ;               // bkj = B(k,j)
                    // scan A(:,k)
                    for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                    {
                        if (!GBB (Ab, pA)) continue ;
                        int64_t i = GBI (Ai, pA, avlen) ;  // get A(i,k)

                        GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                        int8_t f ;

                        #if GB_IS_ANY_MONOID

                            //--------------------------------------------------
                            // ANY monoid
                            //--------------------------------------------------

                            // lock state (3) not needed
                            // 0: not seen: update with new value, f becomes 2
                            // 1: masked, do nothing, f stays 1
                            // 2: already updated, do nothing, f stays 2
                            // 3: state not used, f can be 2
                            GB_ATOMIC_READ
                            f = Hf [i] ;
                            if (!f)
                            {
                                GB_ATOMIC_WRITE
                                Hf [i] = 2 ;
                                GB_ATOMIC_WRITE_HX (i, t) ;    // Hx [i] = t
                            }


                        #else

                        GB_ATOMIC_READ
                        f = Hf [i] ;            // grab the entry
                        #if GB_HAS_ATOMIC
                        if (f == 2)             // if true, update C(i,j)
                        {   GB_cov[456]++ ;
// NOT COVERED (456):
GB_GOTCHA ;
                            GB_ATOMIC_UPDATE_HX (i, t) ;   // Hx [i] += t
                            continue ;          // C(i,j) has been updated
                        }
                        #endif
                        if (f == 1) continue ; // M(i,j)=1; ignore C(i,j)
                        do  // lock the entry
                        {   GB_cov[457]++ ;
// covered (457): 131
                            // do this atomically:
                            // { f = Hf [i] ; Hf [i] = 3 ; }
                            GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;
                        } while (f == 3) ; // lock owner of gets f=0 or 2
                        if (f == 0)
                        {   GB_cov[458]++ ;
// covered (458): 104
                            // C(i,j) is a new entry
                            GB_ATOMIC_WRITE_HX (i, t) ;    // Hx [i] = t
                        }
                        else // f == 2
                        {   GB_cov[459]++ ;
// covered (459): 27
                            // C(i,j) already seen
                            GB_ATOMIC_UPDATE_HX (i, t) ;   // Hx [i] += t
                        }
                        GB_ATOMIC_WRITE
                        Hf [i] = 2 ;                // unlock the entry
                        #endif
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // phase2: fine hash task
            //------------------------------------------------------------------

            // Each hash entry Hf [hash] splits into two parts, (h,f).  f
            // is in the 2 least significant bits.  h is 62 bits, and is
            // the 1-based index i of the C(i,j) entry stored at that
            // location in the hash table.

            // If M is present (M or !M), and M(i,j)=1, then (i+1,1)
            // has been inserted into the hash table, in phase0.

            // Given Hf [hash] split into (h,f)

            // h == 0, f == 0: unlocked and unoccupied.
            //                  note that if f=0, h must be zero too.

            // h == i+1, f == 1: unlocked, occupied by M(i,j)=1.
            //                  C(i,j) has not been seen, or is ignored.
            //                  Hx is not initialized.  M is present.
            //                  if !M: this entry will be ignored in C.

            // h == i+1, f == 2: unlocked, occupied by C(i,j).
            //                  Hx is initialized.  M is no longer
            //                  relevant.

            // h == (anything), f == 3: locked.

            int64_t *GB_RESTRICT
                Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
            int64_t hash_bits = (hash_size-1) ;

            if (M == NULL || ignore_mask)
            {   GB_cov[460]++ ;
// covered (460): 3572

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)=A*B(:,j)
                //--------------------------------------------------------------

                // no mask present, or mask ignored
                #undef GB_CHECK_MASK_ij
                #include "GB_AxB_saxpy3_fineHash_phase2.c"

            }
            else if (mask_is_M)
            {

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)<M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                GB_GET_M_j ;                // get M(:,j)
                if (M_dense_in_place)
                {
GB_GOTCHA ; // ok

                    // M(:,j) is dense.  M is not scattered into Hf.

                    ASSERT (!Mask_struct || M_is_bitmap) ;
                    #undef  GB_CHECK_MASK_ij
                    #define GB_CHECK_MASK_ij                        \
                        bool mij =                                  \
                            (M_is_bitmap ? Mjb [i] : 1) &&          \
                            (Mask_struct ? 1 : (Mjx [i] != 0)) ;    \
                        if (!mij) continue ;

                    switch (msize)
                    {
                        default:
                        case 1  : GB_cov[461]++ ;  
// NOT COVERED (461):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint8_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 2  : GB_cov[462]++ ;  
// NOT COVERED (462):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint16_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 4  : GB_cov[463]++ ;  
// NOT COVERED (463):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint32_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 8  : GB_cov[464]++ ;  
// NOT COVERED (464):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint64_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 16  : GB_cov[465]++ ;  
// NOT COVERED (465):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint64_t
                            #define M_SIZE 2
                            #undef  GB_CHECK_MASK_ij
                            #define GB_CHECK_MASK_ij                    \
                                bool mij =                              \
                                    (M_is_bitmap ? Mjb [i] : 1) &&      \
                                    (Mask_struct ? 1 :                  \
                                        (Mjx [2*i] != 0) ||             \
                                        (Mjx [2*i+1] != 0)) ;           \
                                if (!mij) continue ;

                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                    }
                }

                // Given Hf [hash] split into (h,f)

                // h == 0  , f == 0 : unlocked, unoccupied. C(i,j) ignored
                // h == i+1, f == 1 : unlocked, occupied by M(i,j)=1.
                //                    C(i,j) has not been seen.
                //                    Hx is not initialized.
                // h == i+1, f == 2 : unlocked, occupied by C(i,j), M(i,j)=1
                //                    Hx is initialized.
                // h == ..., f == 3 : locked.

                // 0 -> 0 : to ignore, if M(i,j)=0
                // 1 -> 3 : to lock, if i seen for first time
                // 2 -> 3 : to lock, if i seen already
                // 3 -> 2 : to unlock; now i has been seen

                GB_GET_M_j_RANGE (16) ;     // get first and last in M(:,j)
                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                {   GB_cov[466]++ ;
// covered (466): 18
                    if (!GBB (Bb, pB)) continue ;
                    int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
                    GB_GET_A_k ;                // get A(:,k)
                    if (aknz == 0) continue ;
                    GB_GET_B_kj ;               // bkj = B(k,j)
                    #define GB_IKJ                                             \
                    {                                                          \
                        GB_MULT_A_ik_B_kj ;      /* t = A(i,k) * B(k,j) */     \
                        int64_t i1 = i + 1 ;     /* i1 = one-based index */    \
                        int64_t i_unlocked = (i1 << 2) + 2 ;  /* (i+1,2) */    \
                        for (GB_HASH (i))        /* find i in hash table */    \
                        {                                                      \
                            int64_t hf ;                                       \
                            GB_ATOMIC_READ                                     \
                            hf = Hf [hash] ;        /* grab the entry */       \
                            if (GB_HAS_ATOMIC && (hf == i_unlocked))           \
                            {                                                  \
                                /* Hx [hash] += t */                           \
                                GB_ATOMIC_UPDATE_HX (hash, t) ;                \
                                break ;     /* C(i,j) has been updated */      \
                            }                                                  \
                            if (hf == 0) break ; /* M(i,j)=0; ignore Cij */    \
                            if ((hf >> 2) == i1) /* if true, i found */        \
                            {                                                  \
                                do /* lock the entry */                        \
                                {                                              \
                                    /* do this atomically: */                  \
                                    /* { hf = Hf [hash] ; Hf [hash] |= 3 ; }*/ \
                                    GB_ATOMIC_CAPTURE_INT64_OR (hf,Hf[hash],3);\
                                } while ((hf & 3) == 3) ; /* own: f=1,2 */     \
                                if ((hf & 3) == 1) /* f == 1 */                \
                                {                                              \
                                    /* C(i,j) is a new entry in C(:,j) */      \
                                    /* Hx [hash] = t */                        \
                                    GB_ATOMIC_WRITE_HX (hash, t) ;             \
                                }                                              \
                                else /* f == 2 */                              \
                                {                                              \
                                    /* C(i,j) already appears in C(:,j) */     \
                                    /* Hx [hash] += t */                       \
                                    GB_ATOMIC_UPDATE_HX (hash, t) ;            \
                                }                                              \
                                GB_ATOMIC_WRITE                                \
                                Hf [hash] = i_unlocked ; /* unlock entry */    \
                                break ;                                        \
                            }                                                  \
                        }                                                      \
                    }
                    GB_SCAN_M_j_OR_A_k ;
                    #undef GB_IKJ
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)<!M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                GB_GET_M_j ;                // get M(:,j)
                if (M_dense_in_place)
                {
GB_GOTCHA ;
                    // M(:,j) is dense.  M is not scattered into Hf.

                    if (Mask_struct && !M_is_bitmap)
                    {   GB_cov[467]++ ;
// NOT COVERED (467):
GB_GOTCHA ;
                        // structural mask, complemented, and not bitmap.
                        // No work to do.
                        continue ;
                    }

                    #undef  GB_CHECK_MASK_ij
                    #define GB_CHECK_MASK_ij                        \
                        bool mij =                                  \
                            (M_is_bitmap ? Mjb [i] : 1) &&          \
                            (Mask_struct ? 1 : (Mjx [i] != 0)) ;    \
                        if (mij) continue ;

                    switch (msize)
                    {
                        default:
                        case 1  : GB_cov[468]++ ;  
// NOT COVERED (468):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint8_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 2  : GB_cov[469]++ ;  
// NOT COVERED (469):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint16_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 4  : GB_cov[470]++ ;  
// NOT COVERED (470):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint32_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 8  : GB_cov[471]++ ;  
// NOT COVERED (471):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint64_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 16  : GB_cov[472]++ ;  
// NOT COVERED (472):
GB_GOTCHA ;
                        {
                            #define M_TYPE uint64_t
                            #define M_SIZE 2
                            #undef  GB_CHECK_MASK_ij
                            #define GB_CHECK_MASK_ij                    \
                                bool mij =                              \
                                    (M_is_bitmap ? Mjb [i] : 1) &&      \
                                    (Mask_struct ? 1 :                  \
                                        (Mjx [2*i] != 0) ||             \
                                        (Mjx [2*i+1] != 0)) ;           \
                                if (mij) continue ;

                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                    }
                }

                // Given Hf [hash] split into (h,f)

                // h == 0  , f == 0 : unlocked and unoccupied.
                // h == i+1, f == 1 : unlocked, occupied by M(i,j)=1.
                //                    C(i,j) is ignored.
                // h == i+1, f == 2 : unlocked, occupied by C(i,j).
                //                    Hx is initialized.

                // h == (anything), f == 3: locked.

                // 1 -> 1 : to ignore, if M(i,j)=1
                // 0 -> 3 : to lock, if i seen for first time
                // 2 -> 3 : to lock, if i seen already
                // 3 -> 2 : to unlock; now i has been seen

                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                {
                    if (!GBB (Bb, pB)) continue ;
                    int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
                    GB_GET_A_k ;                // get A(:,k)
                    if (aknz == 0) continue ;
                    GB_GET_B_kj ;               // bkj = B(k,j)
                    // scan A(:,k)
                    for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                    {
                        if (!GBB (Ab, pA)) continue ;
                        int64_t i = GBI (Ai, pA, avlen) ;  // get A(i,k)
                        GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                        int64_t i1 = i + 1 ;        // i1 = one-based index
                        int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)
                        int64_t i_masked   = (i1 << 2) + 1 ;    // (i+1,1)
                        for (GB_HASH (i))           // find i in hash table
                        {
                            int64_t hf ;
                            GB_ATOMIC_READ
                            hf = Hf [hash] ;        // grab the entry
                            #if GB_HAS_ATOMIC
                            if (hf == i_unlocked)  // if true, update C(i,j)
                            {   GB_cov[473]++ ;
// covered (473): 1498500
                                GB_ATOMIC_UPDATE_HX (hash, t) ;// Hx [.]+=t
                                break ;         // C(i,j) has been updated
                            }
                            #endif
                            if (hf == i_masked) break ; // M(i,j)=1; ignore
                            int64_t h = (hf >> 2) ;
                            if (h == 0 || h == i1)
                            {
                                // h=0: unoccupied, h=i1: occupied by i
                                do // lock the entry
                                {   GB_cov[474]++ ;
// covered (474): 1521
                                    // do this atomically:
                                    // { hf = Hf [hash] ; Hf [hash] |= 3 ; }
                                    GB_ATOMIC_CAPTURE_INT64_OR (hf,Hf[hash],3) ;
                                } while ((hf & 3) == 3) ; // owner: f=0,1,2
                                if (hf == 0)            // f == 0
                                {   GB_cov[475]++ ;
// covered (475): 1521
                                    // C(i,j) is a new entry in C(:,j)
                                    // Hx [hash] = t
                                    GB_ATOMIC_WRITE_HX (hash, t) ;
                                    GB_ATOMIC_WRITE
                                    Hf [hash] = i_unlocked ; // unlock entry
                                    break ;
                                }
                                if (hf == i_unlocked)   // f == 2
                                {   GB_cov[476]++ ;
// NOT COVERED (476):
GB_GOTCHA ;
                                    // C(i,j) already appears in C(:,j)
                                    // Hx [hash] += t
                                    GB_ATOMIC_UPDATE_HX (hash, t) ;
                                    GB_ATOMIC_WRITE
                                    Hf [hash] = i_unlocked ; // unlock entry
                                    break ;
                                }
                                // hash table occupied, but not with i,
                                // or with i but M(i,j)=1 so C(i,j) ignored
                                GB_ATOMIC_WRITE
                                Hf [hash] = hf ;  // unlock with prior value
                            }
                        }
                    }
                }
            }
        }
    }

// ttt = omp_get_wtime ( ) - ttt ;
// GB_Global_timing_add (9, ttt) ;
// ttt = omp_get_wtime ( ) ;

    //==========================================================================
    // phase3/phase4: count nnz(C(:,j)) for fine tasks, cumsum of Cp
    //==========================================================================

    GB_AxB_saxpy3_cumsum (C, TaskList, nfine, chunk, nthreads) ;

// ttt = omp_get_wtime ( ) - ttt ;
// GB_Global_timing_add (10, ttt) ;
// ttt = omp_get_wtime ( ) ;

    //==========================================================================
    // phase5: numeric phase for coarse tasks, gather for fine tasks
    //==========================================================================

    // allocate Ci and Cx
    int64_t cnz = Cp [cnvec] ;      // ok: C is sparse
    GrB_Info info =
        GB_bix_alloc (C, cnz, false, true, true, Context) ; // ok: C is sparse
    if (info != GrB_SUCCESS)
    {   GB_cov[477]++ ;
// covered (477): 26852
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    int64_t  *GB_RESTRICT Ci = C->i ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;

    #if GB_IS_ANY_PAIR_SEMIRING

        // TODO: create C as a constant-value matrix.

        // ANY_PAIR semiring: result is purely symbolic
        int64_t pC ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (pC = 0 ; pC < cnz ; pC++)
        {   GB_cov[478]++ ;
// covered (478): 6271
            Cx [pC] = GB_CTYPE_CAST (1, 0) ;
        }

        // Just a precaution; these variables are not used below.  Any attempt
        // to access them will lead to a compile error.
        #define Cx is not used
        #define Hx is not used

        // these have been renamed to ANY_PAIR:
        // EQ_PAIR
        // LAND_PAIR
        // LOR_PAIR
        // MAX_PAIR
        // MIN_PAIR
        // TIMES_PAIR

    #endif

// ttt = omp_get_wtime ( ) - ttt ;
// GB_Global_timing_add (11, ttt) ;
// ttt = omp_get_wtime ( ) ;

    bool C_jumbled = false ;
// TODO    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) reduction(||:C_jumbled)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        #if !GB_IS_ANY_PAIR_SEMIRING
        GB_CTYPE *GB_RESTRICT Hx = (GB_CTYPE *) TaskList [taskid].Hx ;
        #endif
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;

        if (taskid < nfine)
        {

            //------------------------------------------------------------------
            // fine task: gather pattern and values
            //------------------------------------------------------------------

            int64_t kk = TaskList [taskid].vector ;
            int team_size = TaskList [taskid].team_size ;
            int leader    = TaskList [taskid].leader ;
            int my_teamid = taskid - leader ;
            int64_t pC = Cp [kk] ;      // ok: C is sparse

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase5: fine Gustavson task, C=A*B, C<M>=A*B, or C<!M>=A*B
                //--------------------------------------------------------------

                // Hf [i] == 2 if C(i,j) is an entry in C(:,j)
                int8_t *GB_RESTRICT
                    Hf = (int8_t *GB_RESTRICT) TaskList [taskid].Hf ;
                int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cvlen, my_teamid, team_size) ;
                if (cjnz == cvlen)
                {
                    // TODO: if all of C is dense, skip this step and
                    // free the pattern of C.
                    // C(:,j) is dense
                    for (int64_t i = istart ; i < iend ; i++)
                    {   GB_cov[479]++ ;
// covered (479): 44227
                        Ci [pC + i] = i ;           // ok: C is sparse
                    }
                    #if !GB_IS_ANY_PAIR_SEMIRING
                    // copy Hx [istart:iend-1] into Cx [pC+istart:pC+iend-1]
                    GB_CIJ_MEMCPY (pC + istart, istart, iend - istart) ;

                    // TODO: if C is a single vector, skip the memcpy of
                    // Hx into Cx.  Instead, free C->x and transplant
                    // C->x = Hx, and do not free Hx.
                    #endif
                }
                else
                {
                    // C(:,j) is sparse
                    pC += TaskList [taskid].my_cjnz ;
                    for (int64_t i = istart ; i < iend ; i++)
                    {
                        if (Hf [i] == 2)
                        {   GB_cov[480]++ ;
// covered (480): 39229
                            GB_CIJ_GATHER (pC, i) ; // Cx [pC] = Hx [i]
                            Ci [pC++] = i ;         // ok: C is sparse
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase5: fine hash task, C=A*B, C<M>=A*B, C<!M>=A*B
                //--------------------------------------------------------------

                // (Hf [hash] & 3) == 2 if C(i,j) is an entry in C(:,j),
                // and the index i of the entry is (Hf [hash] >> 2) - 1.

                int64_t *GB_RESTRICT
                    Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
                int64_t mystart, myend ;
                GB_PARTITION (mystart, myend, hash_size, my_teamid, team_size) ;
                pC += TaskList [taskid].my_cjnz ;
                for (int64_t hash = mystart ; hash < myend ; hash++)
                {
                    int64_t hf = Hf [hash] ;
                    if ((hf & 3) == 2)
                    {   GB_cov[481]++ ;
// covered (481): 982235
                        int64_t i = (hf >> 2) - 1 ; // found C(i,j) in hash
                        Ci [pC] = i ;               // ok: C is sparse
                        GB_CIJ_GATHER (pC, hash) ;  // Cx [pC] = Hx [hash]
                        pC++ ;
                    }
                }
                C_jumbled = true ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // numeric coarse task: compute C(:,kfirst:klast)
            //------------------------------------------------------------------

            int64_t *GB_RESTRICT
                Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
            int64_t kfirst = TaskList [taskid].start ;
            int64_t klast = TaskList [taskid].end ;
            int64_t nk = klast - kfirst + 1 ;
            int64_t mark = 2*nk + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase5: coarse Gustavson task
                //--------------------------------------------------------------

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase5: coarse Gustavson task, C=A*B
                    //----------------------------------------------------------

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_B_j ;                // get B(:,j)
                        #ifdef GB_IDENTITY
                        if (cjnz == cvlen)          // C(:,j) is dense
                        {   GB_cov[482]++ ;
// covered (482): 468516
                            // this requires the monoid identity.  It is not
                            // defined for the generic saxpy3.
                            GB_COMPUTE_DENSE_C_j ;  // C(:,j) = A*B(:,j)
                            continue ;
                        }
                        #endif
                        mark++ ;
                        if (bjnz == 1
                            #if GB_IS_ANY_PAIR_SEMIRING
                            && !A_is_bitmap
                            #endif
                            )
                        {   GB_cov[483]++ ;
// covered (483): 40140
                            // C(:,j) = A(:,k)*B(k,j)
                            if (!GBB (Bb, pB)) continue ;
                            GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE ;
                        }
                        else if (16 * cjnz > cvlen) // C(:,j) is not very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                if (!GBB (Bb, pB)) continue ;
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                // scan A(:,k)
                                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                                {
                                    // get A(i,k)
                                    if (!GBB (Ab, pA)) continue ;
                                    int64_t i = GBI (Ai, pA, avlen) ;
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    if (Hf [i] != mark)
                                    {   GB_cov[484]++ ;
// covered (484): 29028739
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark ;
                                        GB_HX_WRITE (i, t) ;    // Hx [i] = t
                                    }
                                    else
                                    {   GB_cov[485]++ ;
// covered (485): 224617032
                                        // C(i,j) += A(i,k) * B(k,j)
                                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t
                                    }
                                }
                            }
                            GB_GATHER_ALL_C_j(mark) ;   // gather into C(:,j) 
                        }
                        else    // C(:,j) is very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                if (!GBB (Bb, pB)) continue ;
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                // scan A(:,k)
                                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                                {
                                    // get A(i,k)
                                    if (!GBB (Ab, pA)) continue ;
                                    int64_t i = GBI (Ai, pA, avlen) ;
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    if (Hf [i] != mark)
                                    {   GB_cov[486]++ ;
// covered (486): 99844
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark ;
                                        GB_HX_WRITE (i, t) ; // Hx [i] = t
                                        Ci [pC++] = i ;      // ok: C is sparse
                                    }
                                    else
                                    {   GB_cov[487]++ ;
// covered (487): 1129
                                        // C(i,j) += A(i,k) * B(k,j)
                                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t
                                    }
                                }
                            }
                            GB_SORT_AND_GATHER_C_j ;    // gather into C(:,j)
                        }
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase5: coarse Gustavson task, C<M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.

                    // Hf [i] < mark    : M(i,j)=0, C(i,j) is ignored.
                    // Hf [i] == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // Hf [i] == mark+1 : M(i,j)=1, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_B_j ;                // get B(:,j)
                        #ifdef GB_IDENTITY
                        if (cjnz == cvlen)          // C(:,j) is dense
                        {   GB_cov[488]++ ;
// covered (488): 2582
                            // this requires the monoid identity.  It is not
                            // defined for the generic saxpy3.
                            GB_COMPUTE_DENSE_C_j ;  // C(:,j) = A*B(:,j)
                            continue ;
                        }
                        #endif
                        GB_GET_M_j ;            // get M(:,j)
                        GB_GET_M_j_RANGE (64) ; // get first and last in M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        // scatter M(:,j)
                        GB_SCATTER_M_j (pM_start, pM_end, mark) ;
                        if (16 * cjnz > cvlen)  // C(:,j) is not very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {   GB_cov[489]++ ;
// covered (489): 7659291
                                if (!GBB (Bb, pB)) continue ;
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                #define GB_IKJ                                 \
                                {                                              \
                                    int64_t hf = Hf [i] ;                      \
                                    if (hf == mark)                            \
                                    {                                          \
                                        /* C(i,j) = A(i,k) * B(k,j) */         \
                                        Hf [i] = mark1 ;     /* mark as seen */\
                                        GB_MULT_A_ik_B_kj ;  /* t = aik*bkj */ \
                                        GB_HX_WRITE (i, t) ; /* Hx [i] = t */  \
                                    }                                          \
                                    else if (hf == mark1)                      \
                                    {                                          \
                                        /* C(i,j) += A(i,k) * B(k,j) */        \
                                        GB_MULT_A_ik_B_kj ;  /* t = aik*bkj */ \
                                        GB_HX_UPDATE (i, t) ;/* Hx [i] += t */ \
                                    }                                          \
                                }
                                GB_SCAN_M_j_OR_A_k ;
                                #undef GB_IKJ
                            }
                            GB_GATHER_ALL_C_j(mark1) ;  // gather into C(:,j) 
                        }
                        else    // C(:,j) is very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {   GB_cov[490]++ ;
// covered (490): 3220668
                                if (!GBB (Bb, pB)) continue ;
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                #define GB_IKJ                                 \
                                {                                              \
                                    int64_t hf = Hf [i] ;                      \
                                    if (hf == mark)                            \
                                    {                                          \
                                        /* C(i,j) = A(i,k) * B(k,j) */         \
                                        Hf [i] = mark1 ;     /* mark as seen */\
                                        GB_MULT_A_ik_B_kj ;  /* t = aik*bkj */ \
                                        GB_HX_WRITE (i, t) ; /* Hx [i] = t */  \
                                        Ci [pC++] = i ; /* C(:,j) pattern */   \
                                    }                                          \
                                    else if (hf == mark1)                      \
                                    {                                          \
                                        /* C(i,j) += A(i,k) * B(k,j) */        \
                                        GB_MULT_A_ik_B_kj ;  /* t = aik*bkj */ \
                                        GB_HX_UPDATE (i, t) ;/* Hx [i] += t */ \
                                    }                                          \
                                }
                                GB_SCAN_M_j_OR_A_k ;
                                #undef GB_IKJ
                            }
                            GB_SORT_AND_GATHER_C_j ;    // gather into C(:,j)
                        }
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase5: coarse Gustavson task, C<!M>=A*B
                    //----------------------------------------------------------

                    // Since the mask is !M:
                    // Hf [i] < mark    : M(i,j)=0, C(i,j) is not yet seen.
                    // Hf [i] == mark   : M(i,j)=1, so C(i,j) is ignored.
                    // Hf [i] == mark+1 : M(i,j)=0, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_B_j ;                // get B(:,j)
                        #ifdef GB_IDENTITY
                        if (cjnz == cvlen)          // C(:,j) is dense
                        {   GB_cov[491]++ ;
// NOT COVERED (491):
GB_GOTCHA ;
                            // this requires the monoid identity.  It is not
                            // defined for the generic saxpy3.
                            GB_COMPUTE_DENSE_C_j ;  // C(:,j) = A*B(:,j)
                            continue ;
                        }
                        #endif
                        GB_GET_M_j ;            // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        // scatter M(:,j)
                        GB_SCATTER_M_j (pM_start, pM_end, mark) ;
                        if (16 * cjnz > cvlen)  // C(:,j) is not very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                if (!GBB (Bb, pB)) continue ;
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                // scan A(:,k)
                                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                                {
                                    // get A(i,k)
                                    if (!GBB (Ab, pA)) continue ;
                                    int64_t i = GBI (Ai, pA, avlen) ;
                                    int64_t hf = Hf [i] ;
                                    if (hf < mark)
                                    {   GB_cov[492]++ ;
// covered (492): 368
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark1 ;     // mark as seen
                                        GB_MULT_A_ik_B_kj ;  // t =A(i,k)*B(k,j)
                                        GB_HX_WRITE (i, t) ; // Hx [i] = t
                                    }
                                    else if (hf == mark1)
                                    {   GB_cov[493]++ ;
// covered (493): 56
                                        // C(i,j) += A(i,k) * B(k,j)
                                        GB_MULT_A_ik_B_kj ;  // t =A(i,k)*B(k,j)
                                        GB_HX_UPDATE (i, t) ;// Hx [i] += t
                                    }
                                }
                            }
                            GB_GATHER_ALL_C_j(mark1) ;  // gather into C(:,j) 
                        }
                        else    // C(:,j) is very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                if (!GBB (Bb, pB)) continue ;
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                // scan A(:,k)
                                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                                {
                                    // get A(i,k)
                                    if (!GBB (Ab, pA)) continue ;
                                    int64_t i = GBI (Ai, pA, avlen) ;
                                    int64_t hf = Hf [i] ;
                                    if (hf < mark)
                                    {   GB_cov[494]++ ;
// covered (494): 19615
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark1 ;        // mark as seen
                                        GB_MULT_A_ik_B_kj ;  // t =A(i,k)*B(k,j)
                                        GB_HX_WRITE (i, t) ;    // Hx [i] = t
                                        Ci [pC++] = i ; // create C(:,j) pattern
                                    }
                                    else if (hf == mark1)
                                    {   GB_cov[495]++ ;
// covered (495): 31
                                        // C(i,j) += A(i,k) * B(k,j)
                                        GB_MULT_A_ik_B_kj ;  // t =A(i,k)*B(k,j)
                                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t
                                    }
                                }
                            }
                            GB_SORT_AND_GATHER_C_j ;    // gather into C(:,j)
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase5: coarse hash task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;
                int64_t hash_bits = (hash_size-1) ;

                if (M == NULL || ignore_mask)
                {   GB_cov[496]++ ;
// covered (496): 3545

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C=A*B
                    //----------------------------------------------------------

                    // no mask present, or mask ignored (see below)
                    #undef GB_CHECK_MASK_ij
                    #include "GB_AxB_saxpy3_coarseHash_phase5.c"

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    if (M_dense_in_place)
                    {   GB_cov[497]++ ;
// covered (497): 7

                        ASSERT (!Mask_struct || M_is_bitmap) ;
                        #define GB_CHECK_MASK_ij                        \
                            bool mij =                                  \
                                (M_is_bitmap ? Mjb [i] : 1) &&          \
                                (Mask_struct ? 1 : (Mjx [i] != 0)) ;    \
                            if (!mij) continue ;

                        switch (msize)
                        {
                            default:
                            case 1  : GB_cov[498]++ ;  
// covered (498): 7
                            {
                                #define M_TYPE uint8_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 2  : GB_cov[499]++ ;  
// NOT COVERED (499):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint16_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 4  : GB_cov[500]++ ;  
// NOT COVERED (500):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint32_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 8  : GB_cov[501]++ ;  
// NOT COVERED (501):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint64_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 16  : GB_cov[502]++ ;  
// NOT COVERED (502):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint64_t
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                    \
                                    bool mij =                              \
                                        (M_is_bitmap ? Mjb [i] : 1) &&      \
                                        (Mask_struct ? 1 :                  \
                                            (Mjx [2*i] != 0) ||             \
                                            (Mjx [2*i+1] != 0)) ;           \
                                    if (!mij) continue ;

                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                        }
                    }

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark            : M(i,j)=0, C(i,j) is ignored.
                    // h == i, f == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // h == i, f == mark+1 : M(i,j)=1, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_M_j ;                // get M(:,j)
                        GB_GET_M_j_RANGE (64) ;     // get 1st & last in M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;               // hash M(:,j)
                        GB_GET_B_j ;                // get B(:,j)
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {   GB_cov[503]++ ;
// covered (503): 1901
                            if (!GBB (Bb, pB)) continue ;
                            int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
                            GB_GET_B_kj ;               // bkj = B(k,j)
                            #define GB_IKJ                                     \
                            {                                                  \
                                for (GB_HASH (i))       /* find i in hash */   \
                                {                                              \
                                    int64_t f = Hf [hash] ;                    \
                                    if (f < mark) break ; /* M(i,j)=0, ignore*/\
                                    if (Hi [hash] == i)                        \
                                    {                                          \
                                        GB_MULT_A_ik_B_kj ; /* t = aik*bkj */  \
                                        if (f == mark) /* if true, i is new */ \
                                        {                                      \
                                            /* C(i,j) is new */                \
                                            Hf [hash] = mark1 ; /* mark seen */\
                                            GB_HX_WRITE (hash, t) ;/*Hx[.]=t */\
                                            Ci [pC++] = i ;                    \
                                        }                                      \
                                        else                                   \
                                        {                                      \
                                            /* C(i,j) has been seen; update */ \
                                            GB_HX_UPDATE (hash, t) ;           \
                                        }                                      \
                                        break ;                                \
                                    }                                          \
                                }                                              \
                            }
                            GB_SCAN_M_j_OR_A_k ;
                            #undef GB_IKJ
                        }
                        // found i if: Hf [hash] == mark1 and Hi [hash] == i
                        GB_SORT_AND_GATHER_HASHED_C_j (mark1, Hi [hash] == i) ;
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C<!M>=A*B
                    //----------------------------------------------------------

                    if (M_dense_in_place)
                    {
                        // M(:,j) is dense.  M is not scattered into Hf.

                        if (Mask_struct && !M_is_bitmap)
                        {   GB_cov[504]++ ;
// NOT COVERED (504):
GB_GOTCHA ;
                            // structural mask, complemented, not bitmap.
                            // No work to do.
                            continue ;
                        }

                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij                        \
                            bool mij =                                  \
                                (M_is_bitmap ? Mjb [i] : 1) &&          \
                                (Mask_struct ? 1 : (Mjx [i] != 0)) ;    \
                            if (mij) continue ;

                        switch (msize)
                        {
                            default:
                            case 1  : GB_cov[505]++ ;  
// covered (505): 5
                            {
                                #define M_TYPE uint8_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 2  : GB_cov[506]++ ;  
// NOT COVERED (506):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint16_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 4  : GB_cov[507]++ ;  
// NOT COVERED (507):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint32_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 8  : GB_cov[508]++ ;  
// NOT COVERED (508):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint64_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 16  : GB_cov[509]++ ;  
// NOT COVERED (509):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint64_t
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                    \
                                    bool mij =                              \
                                        (M_is_bitmap ? Mjb [i] : 1) &&      \
                                        (Mask_struct ? 1 :                  \
                                            (Mjx [2*i] != 0) ||             \
                                            (Mjx [2*i+1] != 0)) ;           \
                                    if (mij) continue ;

                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                        }
                    }

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark: unoccupied, M(i,j)=0, and C(i,j) not yet seen.
                    // h == i, f == mark   : M(i,j)=1. C(i,j) ignored.
                    // h == i, f == mark+1 : M(i,j)=0, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_M_j ;                // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;               // hash M(:,j)
                        GB_GET_B_j ;                // get B(:,j)
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            if (!GBB (Bb, pB)) continue ;
                            int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
                            GB_GET_B_kj ;               // bkj = B(k,j)
                            // scan A(:,k)
                            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                            {
                                if (!GBB (Ab, pA)) continue ;
                                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)
                                for (GB_HASH (i))       // find i in hash
                                {
                                    int64_t f = Hf [hash] ;
                                    if (f < mark)   // if true, i is new
                                    {   GB_cov[510]++ ;
// covered (510): 33467
                                        // C(i,j) is new
                                        Hf [hash] = mark1 ; // mark C(i,j) seen
                                        Hi [hash] = i ;
                                        GB_MULT_A_ik_B_kj ; // t = A(i,k)*B(k,j)
                                        GB_HX_WRITE (hash, t) ; // Hx [hash] = t
                                        Ci [pC++] = i ;         // ok: C sparse
                                        break ;
                                    }
                                    if (Hi [hash] == i)
                                    {
                                        if (f == mark1)
                                        {   GB_cov[511]++ ;
// covered (511): 46
                                            // C(i,j) has been seen; update it.
                                            GB_MULT_A_ik_B_kj ;//t=A(i,k)*B(k,j)
                                            GB_HX_UPDATE (hash, t) ;//Hx[ ] += t
                                        }
                                        break ;
                                    }
                                }
                            }
                        }
                        // found i if: Hf [hash] == mark1 and Hi [hash] == i
                        GB_SORT_AND_GATHER_HASHED_C_j (mark1, Hi [hash] == i) ;
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // log the state of C->jumbled
    //--------------------------------------------------------------------------

    C->jumbled = C_jumbled ;

// ttt = omp_get_wtime ( ) - ttt ;
// GB_Global_timing_add (12, ttt) ;

}

#undef Cx
#undef Hx

