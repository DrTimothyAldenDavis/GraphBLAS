//------------------------------------------------------------------------------
// GB_AxB_saxpy3_template.h: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Definitions for GB_AxB_saxpy3_template.c

#ifndef GB_AXB_SAXPY3_TEMPLATE_H
#define GB_AXB_SAXPY3_TEMPLATE_H

//------------------------------------------------------------------------------
// GB_GET_M_j: prepare to iterate over M(:,j)
//------------------------------------------------------------------------------

// prepare to iterate over the vector M(:,j), for the (kk)th vector of B
// FUTURE::: lookup all M(:,j) for all vectors in B, in a single pass,
// and save the mapping (like C_to_M mapping in GB_ewise_slice)
#define GB_GET_M_j                                              \
    int64_t mpleft = 0 ;                                        \
    int64_t mpright = mnvec-1 ;                                 \
    int64_t pM_start, pM_end ;                                  \
    GB_lookup (M_is_hyper, Mh, Mp, mvlen, &mpleft, mpright,     \
        GBH (Bh, kk), &pM_start, &pM_end) ;                     \
    int64_t mjnz = pM_end - pM_start ;    /* nnz (M (:,j)) */

//------------------------------------------------------------------------------
// GB_GET_M_j_RANGE
//------------------------------------------------------------------------------

#define GB_GET_M_j_RANGE(gamma)                                 \
    int64_t mjnz_much = mjnz * gamma

//------------------------------------------------------------------------------
// GB_SCATTER_M_j: scatter M(:,j) for a fine or coarse Gustavson task
//------------------------------------------------------------------------------

#define GB_SCATTER_M_j_TYPE(mask_t,pMstart,pMend,mark)                  \
{                                                                       \
    const mask_t *GB_RESTRICT Mxx = (mask_t *) Mx ;                     \
    if (M_is_bitmap)                                                    \
    {                                                                   \
        /* scan M(:,j) */                                               \
        for (int64_t pM = pMstart ; pM < pMend ; pM++)                  \
        {                                                               \
            /* Hf [i] = M(i,j) */                                       \
            if (Mb [pM] && Mxx [pM]) Hf [GBI (Mi, pM, mvlen)] = mark ;  \
        }                                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        /* scan M(:,j) */                                               \
        for (int64_t pM = pMstart ; pM < pMend ; pM++)                  \
        {                                                               \
            /* Hf [i] = M(i,j) */                                       \
            if (Mxx [pM]) Hf [GBI (Mi, pM, mvlen)] = mark ;             \
        }                                                               \
    }                                                                   \
}                                                                       \
break ;

// scatter M(:,j) for a coarse Gustavson task, C<M>=A*B or C<!M>=A*B
#define GB_SCATTER_M_j(pMstart,pMend,mark)                                  \
    if (Mx == NULL)                                                         \
    {                                                                       \
        /* mask is structural, not valued */                                \
        if (M_is_bitmap)                                                    \
        {                                                                   \
            for (int64_t pM = pMstart ; pM < pMend ; pM++)                  \
            {                                                               \
                /* Hf [i] = M(i,j) */                                       \
                if (Mb [pM]) Hf [GBI (Mi, pM, mvlen)] = mark ;              \
            }                                                               \
        }                                                                   \
        else                                                                \
        {                                                                   \
            for (int64_t pM = pMstart ; pM < pMend ; pM++)                  \
            {                                                               \
                /* Hf [i] = M(i,j) */                                       \
                Hf [GBI (Mi, pM, mvlen)] = mark ;                           \
            }                                                               \
        }                                                                   \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* mask is valued, not structural */                                \
        switch (msize)                                                      \
        {                                                                   \
            default:                                                        \
            case 1: GB_SCATTER_M_j_TYPE (uint8_t , pMstart, pMend, mark) ;  \
            case 2: GB_SCATTER_M_j_TYPE (uint16_t, pMstart, pMend, mark) ;  \
            case 4: GB_SCATTER_M_j_TYPE (uint32_t, pMstart, pMend, mark) ;  \
            case 8: GB_SCATTER_M_j_TYPE (uint64_t, pMstart, pMend, mark) ;  \
            case 16:                                                        \
            {                                                               \
                const uint64_t *GB_RESTRICT Mxx = (uint64_t *) Mx ;         \
                /* scan M(:,j) */                                           \
                for (int64_t pM = pMstart ; pM < pMend ; pM++)              \
                {                                                           \
                    if (!GBB (Mb, pM)) continue ;                           \
                    if (Mxx [2*pM] || Mxx [2*pM+1])                         \
                    {                                                       \
                        /* Hf [i] = M(i,j) */                               \
                        int64_t i = GBI (Mi, pM, mvlen) ;                   \
                        Hf [i] = mark ;                                     \
                    }                                                       \
                }                                                           \
            }                                                               \
        }                                                                   \
    }

//------------------------------------------------------------------------------
// GB_HASH_M_j: scatter M(:,j) for a coarse hash task
//------------------------------------------------------------------------------

// hash M(:,j) into Hf and Hi for coarse hash task, C<M>=A*B or C<!M>=A*B
#define GB_HASH_M_j                                                     \
    for (int64_t pM = pM_start ; pM < pM_end ; pM++) /* scan M(:,j) */  \
    {                                                                   \
        GB_GET_M_ij (pM) ;      /* get M(i,j) */                        \
        if (!mij) continue ;    /* skip if M(i,j)=0 */                  \
        int64_t i = GBI (Mi, pM, mvlen) ;                               \
        for (GB_HASH (i))       /* find i in hash */                    \
        {                                                               \
            if (Hf [hash] < mark)                                       \
            {                                                           \
                Hf [hash] = mark ;  /* insert M(i,j)=1 */               \
                Hi [hash] = i ;                                         \
                break ;                                                 \
            }                                                           \
        }                                                               \
    }

//------------------------------------------------------------------------------
// GB_GET_B_j: prepare to iterate over B(:,j)
//------------------------------------------------------------------------------

#if GB_IS_SECONDJ_MULTIPLIER
    #define GB_GET_T_FOR_SECONDJ \
        GB_CIJ_DECLARE (t) ;        /* ctype t ;        */      \
        GB_MULT (t, ignore, ignore, i, k, j)  /* t = aik * bkj ;  */
#else
    #define GB_GET_T_FOR_SECONDJ ;
#endif

// prepare to iterate over the vector B(:,j), the (kk)th vector in B, where 
// j == GBH (Bh, kk).

#define GB_GET_B_j                                                          \
    int64_t pleft = 0 ;                                                     \
    int64_t pright = anvec-1 ;                                              \
    int64_t j = GBH (Bh, kk) ;                                              \
    GB_GET_T_FOR_SECONDJ ;  /* t = j for SECONDJ, or j+1 for SECONDJ1 */    \
    int64_t pB     = GBP (Bp, kk, bvlen) ;                                  \
    int64_t pB_end = GBP (Bp, kk+1, bvlen) ;                                \
    int64_t bjnz = pB_end - pB ;  /* nnz (B (:,j) */                        \
    /* FUTURE::: can skip if mjnz == 0 for C<M>=A*B tasks */                \
    if (A_is_hyper && B_is_sparse_or_hyper && bjnz > 2 && !B_jumbled)       \
    {                                                                       \
        /* trim Ah [0..pright] to remove any entries past last B(:,j), */   \
        /* to speed up GB_lookup in GB_GET_A_k. */                          \
        /* This requires that B is not jumbled */                           \
        GB_bracket_right (GBI (Bi, pB_end-1, bvlen), Ah, 0, &pright) ;      \
    }

//------------------------------------------------------------------------------
// GB_GET_B_kj: get the numeric value of B(k,j)
//------------------------------------------------------------------------------

#if GB_IS_FIRSTJ_MULTIPLIER

    // FIRSTJ or FIRSTJ1 multiplier
    // t = aik * bkj = k or k+1
    #define GB_GET_B_kj \
        GB_CIJ_DECLARE (t) ;        /* ctype t ;        */      \
        GB_MULT (t, ignore, ignore, i, k, j)  /* t = aik * bkj ;  */

#else

    #define GB_GET_B_kj \
        GB_GETB (bkj, Bx, pB)       /* bkj = Bx [pB] */

#endif

//------------------------------------------------------------------------------
// GB_GET_A_k: prepare to iterate over the vector A(:,k)
//------------------------------------------------------------------------------

#define GB_GET_A_k                                                          \
    if (B_jumbled) pleft = 0 ;  /* reuse pleft if B is not jumbled */       \
    int64_t pA_start, pA_end ;                                              \
    GB_lookup (A_is_hyper, Ah, Ap, avlen, &pleft, pright, k,                \
        &pA_start, &pA_end) ;                                               \
    int64_t aknz = pA_end - pA_start ;    /* nnz (A (:,k)) */

//------------------------------------------------------------------------------
// GB_GET_M_ij: get the numeric value of M(i,j)
//------------------------------------------------------------------------------

#define GB_GET_M_ij(pM)                             \
    /* get M(i,j), at Mi [pM] and Mx [pM] */        \
    bool mij = GBB (Mb, pM) && GB_mcast (Mx, pM, msize)

//------------------------------------------------------------------------------
// GB_MULT_A_ik_B_kj: declare t and compute t = A(i,k) * B(k,j)
//------------------------------------------------------------------------------

#if GB_IS_PAIR_MULTIPLIER

    // PAIR multiplier: t is always 1; no numeric work to do to compute t.
    // The LXOR_PAIR and PLUS_PAIR semirings need the value t = 1 to use in
    // their monoid operator, however.
    #define t (GB_CTYPE_CAST (1, 0))
    #define GB_MULT_A_ik_B_kj

#elif ( GB_IS_FIRSTJ_MULTIPLIER || GB_IS_SECONDJ_MULTIPLIER )

    // nothing to do; t = aik*bkj already defined in an outer loop
    #define GB_MULT_A_ik_B_kj

#else

    // typical semiring
    #define GB_MULT_A_ik_B_kj                                       \
        GB_GETA (aik, Ax, pA) ;         /* aik = Ax [pA] ;  */      \
        GB_CIJ_DECLARE (t) ;            /* ctype t ;        */      \
        GB_MULT (t, aik, bkj, i, k, j)  /* t = aik * bkj ;  */

#endif

//------------------------------------------------------------------------------
// GB_COMPUTE_DENSE_C_j: compute C(:,j)=A*B(:,j) when C(:,j) is completely dense
//------------------------------------------------------------------------------

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_COMPUTE_DENSE_C_j                                \
        for (int64_t i = 0 ; i < cvlen ; i++)                   \
        {                                                       \
            Ci [pC + i] = i ;       /* ok: C is sparse */       \
        }

#else

    // typical semiring
    #define GB_COMPUTE_DENSE_C_j                                    \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            Ci [pC + i] = i ;       /* ok: C is sparse */           \
            GB_CIJ_WRITE (pC + i, GB_IDENTITY) ; /* C(i,j)=0 */     \
        }                                                           \
        for ( ; pB < pB_end ; pB++)     /* scan B(:,j) */           \
        {                                                           \
            if (!GBB (Bb, pB)) continue ;                           \
            int64_t k = GBI (Bi, pB, bvlen) ;   /* get B(k,j) */    \
            GB_GET_A_k ;                /* get A(:,k) */            \
            if (aknz == 0) continue ;                               \
            GB_GET_B_kj ;               /* bkj = B(k,j) */          \
            /* scan A(:,k) */                                       \
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)        \
            {                                                       \
                if (!GBB (Ab, pA)) continue ;                       \
                int64_t i = GBI (Ai, pA, avlen) ;   /* get A(i,k) */\
                GB_MULT_A_ik_B_kj ;      /* t = A(i,k)*B(k,j) */    \
                GB_CIJ_UPDATE (pC + i, t) ; /* Cx [pC+i]+=t */      \
            }                                                       \
        }

#endif

//------------------------------------------------------------------------------
// GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE: compute C(:,j) when nnz(B(:,j)) == 1
//------------------------------------------------------------------------------

// C(:,j) = A(:,k)*B(k,j) when there is a single entry in B(:,j)
// The mask must not be present.
#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do,
    // except that this method cannot be used if A is bitmap.
    #define GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE                      \
        ASSERT (!A_is_bitmap) ;                                     \
        int64_t k = GBI (Bi, pB, bvlen) ;       /* get B(k,j) */    \
        GB_GET_A_k ;                /* get A(:,k) */                \
        memcpy (Ci + pC, Ai + pA_start, aknz * sizeof (int64_t)) ;  \
        /* C becomes jumbled if A is jumbled */                     \
        C_jumbled = C_jumbled || A_jumbled ;

#else

    // typical semiring
    #define GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE                      \
        int64_t k = GBI (Bi, pB, bvlen) ;       /* get B(k,j) */    \
        GB_GET_A_k ;                /* get A(:,k) */                \
        GB_GET_B_kj ;               /* bkj = B(k,j) */              \
        /* scan A(:,k) */                                           \
        for (int64_t pA = pA_start ; pA < pA_end ; pA++)            \
        {                                                           \
            if (!GBB (Ab, pA)) continue ;                           \
            int64_t i = GBI (Ai, pA, avlen) ;  /* get A(i,k) */     \
            GB_MULT_A_ik_B_kj ;         /* t = A(i,k)*B(k,j) */     \
            GB_CIJ_WRITE (pC, t) ;      /* Cx [pC] = t */           \
            Ci [pC++] = i ;             /* ok: C is sparse */       \
        }                                                           \
        /* C becomes jumbled if A is jumbled */                     \
        C_jumbled = C_jumbled || A_jumbled ;

#endif

//------------------------------------------------------------------------------
// GB_GATHER_ALL_C_j: gather the values and pattern of C(:,j)
//------------------------------------------------------------------------------

// gather the pattern and values of C(:,j) for a coarse Gustavson task;
// the pattern is not flagged as jumbled.

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_GATHER_ALL_C_j(mark)                                 \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            if (Hf [i] == mark)                                     \
            {                                                       \
                Ci [pC++] = i ;         /* ok: C is sparse */       \
            }                                                       \
        }

#else

    // typical semiring
    #define GB_GATHER_ALL_C_j(mark)                                 \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            if (Hf [i] == mark)                                     \
            {                                                       \
                GB_CIJ_GATHER (pC, i) ; /* Cx [pC] = Hx [i] */      \
                Ci [pC++] = i ;         /* ok: C is sparse */       \
            }                                                       \
        }

#endif

//------------------------------------------------------------------------------
// GB_SORT_AND_GATHER_C_j: gather the values pattern of C(:,j)
//------------------------------------------------------------------------------

// gather the values of C(:,j) for a coarse Gustavson task
#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; just flag the pattern as jumbled
    #define GB_SORT_AND_GATHER_C_j                              \
        /* the pattern of C(:,j) is now jumbled */              \
        C_jumbled = true ;

#else

    // typical semiring
    #define GB_SORT_AND_GATHER_C_j                              \
        /* the pattern of C(:,j) is now jumbled */              \
        C_jumbled = true ;                                      \
        /* gather the values into C(:,j) */                     \
        for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)      \
        {                                                       \
            int64_t i = Ci [pC] ;       /* ok: C is sparse */   \
            GB_CIJ_GATHER (pC, i) ;   /* Cx [pC] = Hx [i] */    \
        }

#endif

//------------------------------------------------------------------------------
// GB_SORT_AND_GATHER_HASHED_C_j: gather values for coarse hash 
//------------------------------------------------------------------------------

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; just flag the pattern as jumbled
    #define GB_SORT_AND_GATHER_HASHED_C_j(hash_mark,Hi_hash_equals_i)       \
        /* the pattern of C(:,j) is now jumbled */                          \
        C_jumbled = true ;

#else

    // gather the values of C(:,j) for a coarse hash task
    #define GB_SORT_AND_GATHER_HASHED_C_j(hash_mark,Hi_hash_equals_i)       \
        /* the pattern of C(:,j) is now jumbled */                          \
        C_jumbled = true ;                                                  \
        for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)                  \
        {                                                                   \
            int64_t i = Ci [pC] ;       /* ok: C is sparse */               \
            int64_t marked = (hash_mark) ;                                  \
            for (GB_HASH (i))           /* find i in hash table */          \
            {                                                               \
                if (Hf [hash] == marked && (Hi_hash_equals_i))              \
                {                                                           \
                    /* i found in the hash table */                         \
                    /* Cx [pC] = Hx [hash] ; */                             \
                    GB_CIJ_GATHER (pC, hash) ;                              \
                    break ;                                                 \
                }                                                           \
            }                                                               \
        }

#endif

//------------------------------------------------------------------------------
// GB_SCAN_M_j_OR_A_k: compute C(:,j) using linear scan or binary search
//------------------------------------------------------------------------------

#if 1
// C(:,j)<M(:,j)>=A(:,k)*B(k,j) using one of two methods
#define GB_SCAN_M_j_OR_A_k                                              \
{                                                                       \
    if (aknz > 256 && mjnz_much < aknz && mjnz < mvlen && aknz < avlen  \
        && !A_jumbled)                                                  \
    {                                                                   \
        /* M and A are both sparse, and nnz(M(:,j)) much less than */   \
        /* nnz(A(:,k)); scan M(:,j), and do binary search for A(i,k).*/ \
        /* This requires that A is not jumbled. */                      \
        int64_t pA = pA_start ;                                         \
        for (int64_t pM = pM_start ; pM < pM_end ; pM++)                \
        {                                                               \
            GB_GET_M_ij (pM) ;      /* get M(i,j) */                    \
            if (!mij) continue ;    /* skip if M(i,j)=0 */              \
            int64_t i = Mi [pM] ;   /* ok: M and A are sparse */        \
            bool found ;            /* search for A(i,k) */             \
            int64_t apright = pA_end - 1 ;                              \
            /* the binary search can only be done if A is not jumbled */\
            GB_BINARY_SEARCH (i, Ai, pA, apright, found) ;              \
            if (found)                                                  \
            {                                                           \
                /* C(i,j)<M(i,j)> += A(i,k) * B(k,j) for this method. */\
                /* M(i,j) is now known to be equal to 1, so there are */\
                /* cases in the GB_IKJ operation that can never */      \
                /* occur.  This could be pruned from the GB_IKJ */      \
                /* operation, but then this operation would differ */   \
                /* from the GB_IKJ operation in the linear-time scan */ \
                /* of A(:,j), below.  It's unlikely that pruning this */\
                /* case would lead to much performance improvement. */  \
                GB_IKJ ;                                                \
            }                                                           \
        }                                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        /* A(:,j) is sparse enough relative to M(:,j) */                \
        /* M and/or A can dense, and either can be jumbled. */          \
        /* scan A(:,k), and lookup M(i,j) (in the hash table) */        \
        for (int64_t pA = pA_start ; pA < pA_end ; pA++)                \
        {                                                               \
            if (!GBB (Ab, pA)) continue ;                               \
            int64_t i = GBI (Ai, pA, avlen) ;    /* get A(i,k) */       \
            /* do C(i,j)<M(i,j)> += A(i,k) * B(k,j) for this method */  \
            /* M(i,j) may be 0 or 1, as given in the hash table */      \
            GB_IKJ ;                                                    \
        }                                                               \
    }                                                                   \
}
#endif

#if 0
// C(:,j)<M(:,j)>=A(:,k)*B(k,j)
#define GB_SCAN_M_j_OR_A_k                                              \
{                                                                       \
    {                                                                   \
        /* A(:,j) is sparse enough relative to M(:,j) */                \
        /* M and/or A can dense, and either can be jumbled. */          \
        /* scan A(:,k), and lookup M(i,j) (in the hash table) */        \
        for (int64_t pA = pA_start ; pA < pA_end ; pA++)                \
        {                                                               \
            if (!GBB (Ab, pA)) continue ;                               \
            int64_t i = GBI (Ai, pA, avlen) ;    /* get A(i,k) */       \
            /* do C(i,j)<M(i,j)> += A(i,k) * B(k,j) for this method */  \
            /* M(i,j) may be 0 or 1, as given in the hash table */      \
            GB_IKJ ;                                                    \
        }                                                               \
    }                                                                   \
}
#endif

//------------------------------------------------------------------------------
// GB_ATOMIC_UPDATE_HX:  Hx [i] += t
//------------------------------------------------------------------------------

#if GB_IS_ANY_MONOID

    //--------------------------------------------------------------------------
    // The update Hx [i] += t can be skipped entirely, for the ANY monoid.
    //--------------------------------------------------------------------------

    #define GB_ATOMIC_UPDATE_HX(i,t)

#elif GB_HAS_ATOMIC

    //--------------------------------------------------------------------------
    // Hx [i] += t via atomic update
    //--------------------------------------------------------------------------

    // for built-in MIN/MAX monoids only, on built-in types
    #define GB_MINMAX(i,t,done)                                     \
    {                                                               \
        GB_CTYPE xold, xnew, *px = Hx + (i) ;                       \
        do                                                          \
        {                                                           \
            /* xold = Hx [i] via atomic read */                     \
            GB_ATOMIC_READ                                          \
            xold = (*px) ;                                          \
            /* done if xold <= t for MIN, or xold >= t for MAX, */  \
            /* but not done if xold is NaN */                       \
            if (done) break ;                                       \
            xnew = t ;  /* t should be assigned; it is not NaN */   \
        }                                                           \
        while (!GB_ATOMIC_COMPARE_EXCHANGE (px, xold, xnew)) ;      \
    }

    #if GB_IS_IMIN_ATOMIC

        // built-in MIN monoids for signed and unsigned integers
        #define GB_ATOMIC_UPDATE_HX(i,t)                            \
            GB_MINMAX (i, t, xold <= t)

    #elif GB_IS_IMAX_ATOMIC

        // built-in MAX monoids for signed and unsigned integers
        #define GB_ATOMIC_UPDATE_HX(i,t)                            \
            GB_MINMAX (i, t, xold >= t)

    #elif GB_IS_FMIN_ATOMIC

        // built-in MIN monoids for float and double, with omitnan behavior.
        // The update is skipped entirely if t is NaN.  Otherwise, if t is not
        // NaN, xold is checked.  If xold is NaN, islessequal (xold, t) is
        // always false, so the non-NaN t must be always be assigned to Hx [i].
        // If both terms are not NaN, then islessequal (xold,t) is just the
        // comparison xold <= t.  If that is true, there is no work to do and
        // the loop breaks.  Otherwise, t is smaller than xold and so it must
        // be assigned to Hx [i].
        #define GB_ATOMIC_UPDATE_HX(i,t)                            \
        {                                                           \
            if (!isnan (t))                                         \
            {                                                       \
                GB_MINMAX (i, t, islessequal (xold, t)) ;           \
            }                                                       \
        }

    #elif GB_IS_FMAX_ATOMIC

        // built-in MAX monoids for float and double, with omitnan behavior.
        #define GB_ATOMIC_UPDATE_HX(i,t)                            \
        {                                                           \
            if (!isnan (t))                                         \
            {                                                       \
                GB_MINMAX (i, t, isgreaterequal (xold, t)) ;        \
            }                                                       \
        }

    #elif GB_IS_PLUS_FC32_MONOID

        // built-in PLUS_FC32 monoid
        #define GB_ATOMIC_UPDATE_HX(i,t)                            \
            GB_ATOMIC_UPDATE                                        \
            Hx_real [2*(i)] += crealf (t) ;                         \
            GB_ATOMIC_UPDATE                                        \
            Hx_imag [2*(i)] += cimagf (t) ;

    #elif GB_IS_PLUS_FC64_MONOID

        // built-in PLUS_FC64 monoid
        #define GB_ATOMIC_UPDATE_HX(i,t)                            \
            GB_ATOMIC_UPDATE                                        \
            Hx_real [2*(i)] += creal (t) ;                          \
            GB_ATOMIC_UPDATE                                        \
            Hx_imag [2*(i)] += cimag (t) ;

    #elif GB_HAS_OMP_ATOMIC

        // built-in PLUS and TIMES for integers and real, and boolean LOR,
        // LAND, LXOR monoids can be implemented with an OpenMP pragma.
        #define GB_ATOMIC_UPDATE_HX(i,t)                            \
            GB_ATOMIC_UPDATE                                        \
            GB_HX_UPDATE (i, t)

    #else

        // all other atomic monoids (boolean EQ, bitwise monoids, etc)
        // on boolean, signed and unsigned integers, float, and double
        // (not used for single and double complex).
        #define GB_ATOMIC_UPDATE_HX(i,t)                            \
        {                                                           \
            GB_CTYPE xold, xnew, *px = Hx + (i) ;                   \
            do                                                      \
            {                                                       \
                /* xold = Hx [i] via atomic read */                 \
                GB_ATOMIC_READ                                      \
                xold = (*px) ;                                      \
                /* xnew = xold + t */                               \
                xnew = GB_ADD_FUNCTION (xold, t) ;                  \
            }                                                       \
            while (!GB_ATOMIC_COMPARE_EXCHANGE (px, xold, xnew)) ;  \
        }

    #endif

#else

    //--------------------------------------------------------------------------
    // Hx [i] += t can only be done inside the critical section
    //--------------------------------------------------------------------------

    // all user-defined monoids go here, and all complex monoids (except PLUS)
    #define GB_ATOMIC_UPDATE_HX(i,t)    \
        GB_OMP_FLUSH                    \
        GB_HX_UPDATE (i, t) ;           \
        GB_OMP_FLUSH

#endif

#define GB_IS_MINMAX_MONOID \
    (GB_IS_IMIN_MONOID || GB_IS_IMAX_MONOID ||  \
     GB_IS_FMIN_MONOID || GB_IS_FMAX_MONOID)

//------------------------------------------------------------------------------
// GB_ATOMIC_WRITE_HX:  Hx [i] = t
//------------------------------------------------------------------------------

#if GB_IS_ANY_PAIR_SEMIRING

    //--------------------------------------------------------------------------
    // ANY_PAIR: result is purely symbolic; no numeric work to do
    //--------------------------------------------------------------------------

    #define GB_ATOMIC_WRITE_HX(i,t)

#elif GB_HAS_ATOMIC

    //--------------------------------------------------------------------------
    // Hx [i] = t via atomic write
    //--------------------------------------------------------------------------

    #if GB_IS_PLUS_FC32_MONOID

        // built-in PLUS_FC32 monoid
        #define GB_ATOMIC_WRITE_HX(i,t)                             \
            GB_ATOMIC_WRITE                                         \
            Hx_real [2*(i)] = crealf (t) ;                          \
            GB_ATOMIC_WRITE                                         \
            Hx_imag [2*(i)] = cimagf (t) ;

    #elif GB_IS_PLUS_FC64_MONOID

        // built-in PLUS_FC64 monoid
        #define GB_ATOMIC_WRITE_HX(i,t)                             \
            GB_ATOMIC_WRITE                                         \
            Hx_real [2*(i)] = creal (t) ;                           \
            GB_ATOMIC_WRITE                                         \
            Hx_imag [2*(i)] = cimag (t) ;

    #else

        // all other atomic monoids
        #define GB_ATOMIC_WRITE_HX(i,t)                             \
            GB_ATOMIC_WRITE                                         \
            GB_HX_WRITE (i, t)

    #endif

#else

    //--------------------------------------------------------------------------
    // Hx [i] = t via critical section
    //--------------------------------------------------------------------------

    #define GB_ATOMIC_WRITE_HX(i,t)     \
        GB_OMP_FLUSH                    \
        GB_HX_WRITE (i, t) ;            \
        GB_OMP_FLUSH

#endif

//------------------------------------------------------------------------------
// hash iteration
//------------------------------------------------------------------------------

// to iterate over the hash table, looking for index i:
// 
//      for (GB_HASH (i))
//      {
//          ...
//      }
//
// which expands into the following, where f(i) is the GB_HASHF(i) hash
// function:
//
//      for (int64_t hash = f(i) ; ; hash = (hash+1)&(hash_size-1))
//      {
//          ...
//      }

#define GB_HASH(i) \
    int64_t hash = GB_HASHF (i) ; ; GB_REHASH (hash,i)

#endif

