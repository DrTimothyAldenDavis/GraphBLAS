//------------------------------------------------------------------------------
// GB_AxB:  hard-coded functions for semiring: C<M>=A*B or A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated).

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "GB_ek_slice.h"
#include "GB_bracket.h"
#include "GB_sort.h"
#include "GB_atomics.h"
#include "GB_AxB_saxpy3.h"
#include "GB_AxB__include.h"
#include "GB_unused.h"

// The C=A*B semiring is defined by the following types and operators:

// A'*B function (dot2):     GB_Adot2B__any_firstj_int32
// A'*B function (dot3):     GB_Adot3B__any_firstj_int32
// C+=A'*B function (dot4):  GB_Adot4B__any_firstj_int32
// A*B function (saxpy3):    GB_Asaxpy3B__any_firstj_int32

// C type:   int32_t
// A type:   int32_t
// B type:   int32_t

// Multiply: z = k
// Add:      cij = z
//           'any' monoid?  1
//           atomic?        1
//           OpenMP atomic? 0
// MultAdd:  cij = k
// Identity: 0
// Terminal: break ;

#define GB_ATYPE \
    int32_t

#define GB_BTYPE \
    int32_t

#define GB_CTYPE \
    int32_t

#define GB_CSIZE (sizeof (GB_CTYPE))

// true for int64, uint64, float, double, float complex, and double complex 
#define GB_CTYPE_IGNORE_OVERFLOW \
    0

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA) \
    ;

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB) \
    ;

#define GB_CX(p) Cx [p]

// multiply operator
#define GB_MULT(z, x, y, i, k, j) \
    z = k

// cast from a real scalar (or 2, if C is complex) to the type of C
#define GB_CTYPE_CAST(x,y) \
    ((int32_t) x)

// multiply-add
#define GB_MULTADD(z, x, y, i, k, j) \
    z = k

// monoid identity value
#define GB_IDENTITY \
    0

// break if cij reaches the terminal value (dot product only)
#define GB_DOT_TERMINAL(cij) \
    break ;

// simd pragma for dot-product loop vectorization
#define GB_PRAGMA_SIMD_DOT(cij) \
    ;

// simd pragma for other loop vectorization
#define GB_PRAGMA_SIMD_VECTORIZE GB_PRAGMA_SIMD

// 1 for the PLUS_PAIR_(real) semirings, not for the complex case
#define GB_IS_PLUS_PAIR_REAL_SEMIRING \
    0

// declare the cij scalar
#if GB_IS_PLUS_PAIR_REAL_SEMIRING
    // also initialize cij to zero
    #define GB_CIJ_DECLARE(cij) \
        int32_t cij = 0
#else
    // all other semirings: just declare cij, do not initialize it
    #define GB_CIJ_DECLARE(cij) \
        int32_t cij
#endif

// cij = Cx [pC]
#define GB_GETC(cij,p) cij = Cx [p]

// Cx [pC] = cij
#define GB_PUTC(cij,p) Cx [p] = cij

// Cx [p] = t
#define GB_CIJ_WRITE(p,t) Cx [p] = t

// C(i,j) += t
#define GB_CIJ_UPDATE(p,t) \
    Cx [p] = t

// x + y
#define GB_ADD_FUNCTION(x,y) \
    y

// bit pattern for bool, 8-bit, 16-bit, and 32-bit integers
#define GB_CTYPE_BITS \
    0xffffffffL

// 1 if monoid update can skipped entirely (the ANY monoid)
#define GB_IS_ANY_MONOID \
    1

// 1 if monoid update is EQ
#define GB_IS_EQ_MONOID \
    0

// 1 if monoid update can be done atomically, 0 otherwise
#define GB_HAS_ATOMIC \
    1

// 1 if monoid update can be done with an OpenMP atomic update, 0 otherwise
#if GB_MICROSOFT
    #define GB_HAS_OMP_ATOMIC \
        0
#else
    #define GB_HAS_OMP_ATOMIC \
        0
#endif

// 1 for the ANY_PAIR semirings
#define GB_IS_ANY_PAIR_SEMIRING \
    0

// 1 if PAIR is the multiply operator 
#define GB_IS_PAIR_MULTIPLIER \
    0

// 1 if monoid is PLUS_FC32
#define GB_IS_PLUS_FC32_MONOID \
    0

// 1 if monoid is PLUS_FC64
#define GB_IS_PLUS_FC64_MONOID \
    0

// 1 for the FIRSTI or FIRSTI1 multiply operator
#define GB_IS_FIRSTI_MULTIPLIER \
    0

// 1 for the FIRSTJ or FIRSTJ1 multiply operator
#define GB_IS_FIRSTJ_MULTIPLIER \
    1

// 1 for the SECONDJ or SECONDJ1 multiply operator
#define GB_IS_SECONDJ_MULTIPLIER \
    0

// atomic compare-exchange
#define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
    GB_ATOMIC_COMPARE_EXCHANGE_32 (target, expected, desired)

#if GB_IS_ANY_PAIR_SEMIRING

    // result is purely symbolic; no numeric work to do.  Hx is not used.
    #define GB_HX_WRITE(i,t)
    #define GB_CIJ_GATHER(p,i)
    #define GB_HX_UPDATE(i,t)
    #define GB_CIJ_MEMCPY(p,i,len)

#else

    // Hx [i] = t
    #define GB_HX_WRITE(i,t) Hx [i] = t

    // Cx [p] = Hx [i]
    #define GB_CIJ_GATHER(p,i) Cx [p] = Hx [i]

    // Hx [i] += t
    #define GB_HX_UPDATE(i,t) \
        Hx [i] = t

    // memcpy (&(Cx [p]), &(Hx [i]), len)
    #define GB_CIJ_MEMCPY(p,i,len) \
        memcpy (Cx +(p), Hx +(i), (len) * sizeof(int32_t))

#endif

// disable this semiring and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_ANY || GxB_NO_FIRSTJ || GxB_NO_INT32 || GxB_NO_ANY_INT32 || GxB_NO_FIRSTJ_INT32 || GxB_NO_ANY_FIRSTJ_INT32)

//------------------------------------------------------------------------------
// C=A'*B, C<M>=A'*B, or C<!M>=A'*B: dot product method where C is bitmap
//------------------------------------------------------------------------------

GrB_Info GB_Adot2B__any_firstj_int32
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const GrB_Matrix A, bool A_is_pattern, int64_t *GB_RESTRICT A_slice,
    const GrB_Matrix B, bool B_is_pattern, int64_t *GB_RESTRICT B_slice,
    int nthreads, int naslice, int nbslice
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
//     #include "GB_AxB_dot2_meta.c"



//------------------------------------------------------------------------------
// GB_AxB_dot2_meta: C=A'*B, C<M>=A'*B or C<!M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// TODO: rename to GB_bitmap_AxB_dot_meta.c

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    int64_t cnvals = 0 ;

    ASSERT (GB_IS_BITMAP (C)) ;
    int8_t   *GB_RESTRICT Cb = C->b ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) (B_is_pattern ? NULL : B->x) ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) (A_is_pattern ? NULL : A->x) ;
    const int64_t cvlen = C->vlen ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int8_t  *GB_RESTRICT Bb = B->b ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const int64_t bnvec = B->nvec ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int8_t  *GB_RESTRICT Ab = A->b ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;

    const int64_t vlen = A->vlen ;
    ASSERT (A->vlen == B->vlen) ;

    const int ntasks = naslice * nbslice ;

    //--------------------------------------------------------------------------
    // C=A'*B, C<M>=A'*B, or C<!M>=A'*B via dot products
    //--------------------------------------------------------------------------

    if (M == NULL)
    { 

        // C = A'*B via dot products
        #include "GB_AxB_dot2_template.c"

    }
    else
    { 

        //----------------------------------------------------------------------
        // get M
        //----------------------------------------------------------------------

        // C<M>=A'*B or C<!M>=A'*B via dot products
        #define GB_MASK_IS_PRESENT
        // #include "GB_AxB_dot2_template.c"

//------------------------------------------------------------------------------
// GB_AxB_dot2_tempate:  C=A'B, C<!M>=A'*B, or C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// TODO: rename GB_bitmap_AxB_dot_template.c

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    #if defined ( GB_MASK_IS_PRESENT )
    // TODO: if M is sparse, scatter into the C bitmap instead
    const int64_t *GB_RESTRICT Mp = M->p ;
    const int64_t *GB_RESTRICT Mh = M->h ;
    const int8_t  *GB_RESTRICT Mb = M->b ;
    const int64_t *GB_RESTRICT Mi = M->i ;
    const GB_void *GB_RESTRICT Mx ;
    Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
    size_t msize = M->type->size ;
    const int64_t mnvec = M->nvec ;
    const int64_t mvlen = M->vlen ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const bool M_is_bitmap_or_full = M_is_bitmap || M_is_full ;
    const bool M_is_bitmap_struct = M_is_bitmap && Mask_struct ;
    #endif

    //--------------------------------------------------------------------------
    // C=A'*B, C<M>=A'*B, or C<!M>=A'*B where C is bitmap
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int a_tid = tid / nbslice ;
        int b_tid = tid % nbslice ;
        int64_t kA_start = A_slice [a_tid] ;
        int64_t kA_end   = A_slice [a_tid+1] ;
        int64_t kB_start = B_slice [b_tid] ;
        int64_t kB_end   = B_slice [b_tid+1] ;
        bool mdense = false ;

        //----------------------------------------------------------------------
        // C=A'*B, C<M>=A'*B, or C<!M>=A'*B via dot products
        //----------------------------------------------------------------------

        for (int64_t kB = kB_start ; kB < kB_end ; kB++)
        {

            //------------------------------------------------------------------
            // get B(:,j) and C(:,j)
            //------------------------------------------------------------------

            int64_t j = GBH (Bh, kB) ;
            int64_t pB_start = GBP (Bp, kB, vlen) ;
            int64_t pB_end   = GBP (Bp, kB+1, vlen) ;
            int64_t bjnz = pB_end - pB_start ;
            // no work to do if B(:,j) is empty
            if (bjnz == 0) continue ;
            int64_t pC_start = j * cvlen ;

            //------------------------------------------------------------------
            // get M(:,j), if present
            //------------------------------------------------------------------

            #if defined ( GB_MASK_IS_PRESENT )
            // TODO: delete this and scatter M into the C bitmap if sparse,
            // or use in-place is M is dense, bitmap, or full
            // find vector j in M
            int64_t pM, pM_end ;
            if (M_is_bitmap_or_full)
            {
                pM = pC_start ;
            }
            else
            {
                int64_t mpleft = 0 ;
                GB_lookup (M_is_hyper, Mh, Mp, mvlen, &mpleft, mnvec-1, j,
                    &pM, &pM_end) ;
                int64_t mjnz = pM_end - pM ;
                mdense = (mjnz == mvlen) ;
            }
            #endif

            //------------------------------------------------------------------
            // C(:,j)<#M(:,j)> = A'*B(:,j), or C(:,j) = A'*B(:,j) if no mask
            //------------------------------------------------------------------

            // get the first and last index in B(:,j)
            int64_t ib_first = GBI (Bi, pB_start, vlen) ;
            int64_t ib_last  = GBI (Bi, pB_end-1, vlen) ;

            for (int64_t kA = kA_start ; kA < kA_end ; kA++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                // int64_t i = GBH (Ah, kA) ;
                int64_t i = kA ; // GBH (Ah, kA) ;

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                #if defined ( GB_MASK_IS_PRESENT )


                bool mij ;

                    mij = Mb [pM + i] ;
#if 0
                if (M_is_bitmap_struct)
                {
                    mij = Mb [pM + i] ;
                }
                else if (M_is_bitmap_or_full)
                {
                    mij = GBB (Mb, pM + i) && GB_mcast (Mx, pM + i, msize) ;
                }
                else 
#endif
#if 0
                if (mdense)
                { 
                    // TODO: do not use this; scatter M into C bitmap instead
                    // M(:,j) is sparse/hyper, but
                    // with a fully-populated vector M(:,j)
                    mij = GB_mcast (Mx, pM + i, msize) ;
                }
                else
#endif
#if 0
                {
                    // M(:,j) is sparse:
                    // TODO: delete this and scatter M into the C bitmap
                    // instead.
                    bool found ;
                    int64_t pright = pM_end - 1 ;
                    GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;
                    mij = found && GB_mcast (Mx, pM, msize) ;
                }
#endif

                // if (mij ^ Mask_comp)
                if (!mij)
                #endif
                { 

                    //----------------------------------------------------------
                    // C(i,j) = A(:,i)'*B(:,j)
                    //----------------------------------------------------------

                    int64_t pA     = Ap [kA] ; // GBP (Ap, kA, vlen) ;
                    int64_t pA_end = Ap [kA+1] ; // GBP (Ap, kA+1, vlen) ;
//                    #include "GB_AxB_dot_cij.c"

//------------------------------------------------------------------------------
// GB_AxB_dot_cij: compute C(i,j) = A(:,i)'*B(:,j)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// computes C(i,j) = A (:,i)'*B(:,j) via sparse dot product.  This template is
// used for all three cases: C=A'*B, C<M>=A'B, and C<!M>=A'*B in dot2, and
// C<M>=A'*B in dot3.  GB_AxB_dot2 does not define GB_DOT3, and constructs
// C as bitmap.  GB_AxB_dot3 defines GB_DOT3 and constructs C as sparse
// with the same pattern as M.

// When used as the multiplicative operator, the PAIR operator provides some
// useful special cases.  Its output is always one, for any matching pair of
// entries A(k,i)'*B(k,j) for some k.  If the monoid is ANY, then C(i,j)=1 if
// the intersection for the dot product is non-empty.  This intersection has to
// be found, in general.  However, suppose B(:,j) is dense.  Then every entry
// in the pattern of A(:,i)' will produce a 1 from the PAIR operator.  If the
// monoid is ANY, then C(i,j)=1 if A(:,i)' is nonempty.  If the monoid is PLUS,
// then C(i,j) is simply nnz(A(:,i)), assuming no overflow.  The XOR monoid
// acts like a 1-bit summation, so the result of the XOR_PAIR_BOOL semiring
// will be C(i,j) = mod (nnz(A(:,i)'*B(:,j)),2).

// If both A(:,i) and B(:,j) are sparse, then the intersection must still be
// found, so these optimizations can be used only if A(:,i) and/or B(:,j) are
// entirely populated.

// For built-in, pre-generated semirings, the PAIR operator is only coupled
// with either the ANY, PLUS, EQ, or XOR monoids, since the other monoids are
// equivalent to the ANY monoid.  With no accumulator, EQ_PAIR is the same as
// ANY_PAIR, they differ for the C+=A'*B operation (see *dot4*).

#include "GB_unused.h"

//------------------------------------------------------------------------------
// GB_DOT: cij += A(k,i) * B(k,j), then break if terminal
//------------------------------------------------------------------------------

// Ai [pA] and Bi [pB] are both equal to the index k.

// use the boolean flag cij_exists to set/check if C(i,j) exists
#undef  GB_CIJ_CHECK
#define GB_CIJ_CHECK true
#undef  GB_CIJ_EXIST
#define GB_CIJ_EXISTS cij_exists
#undef  GB_DOT

#if GB_IS_PLUS_PAIR_REAL_SEMIRING

    //--------------------------------------------------------------------------
    // plus_pair_real semiring
    //--------------------------------------------------------------------------

    #if GB_CTYPE_IGNORE_OVERFLOW

        // PLUS_PAIR for 64-bit integers, float, and double (not complex):
        // To check if C(i,j) exists, test (cij != 0) when done.  The
        // boolean flag cij_exists is not defined.
        #undef  GB_CIJ_CHECK
        #define GB_CIJ_CHECK false
        #undef  GB_CIJ_EXISTS
        #define GB_CIJ_EXISTS (cij != 0)
        #define GB_DOT(k,pA,pB) cij++ ;

    #else

        // PLUS_PAIR semiring for small integers
        #define GB_DOT(k,pA,pB)                                         \
            cij_exists = true ;                                         \
            cij++ ;

    #endif

#elif GB_IS_ANY_MONOID

    //--------------------------------------------------------------------------
    // ANY monoid
    //--------------------------------------------------------------------------

    #if defined ( GB_DOT3 )

        #define GB_DOT(k,pA,pB)                                         \
        {                                                               \
            GB_GETA (aki, Ax, pA) ;  /* aki = A(k,i) */                 \
            GB_GETB (bkj, Bx, pB) ;  /* bkj = B(k,j) */                 \
            /* cij = (A')(i,k) * B(k,j), and add to the pattern */      \
            cij_exists = true ;                                         \
            GB_MULT (cij, aki, bkj, i, k, j) ;                          \
            break ;                                                     \
        }

    #else

        #define GB_DOT(k,pA,pB)                                         \
        {                                                               \
            GB_GETA (aki, Ax, pA) ;  /* aki = A(k,i) */                 \
            GB_GETB (bkj, Bx, pB) ;  /* bkj = B(k,j) */                 \
            /* cij = (A')(i,k) * B(k,j), and add to the pattern */      \
            GB_MULT (cij, aki, bkj, i, k, j) ;                          \
            int64_t pC = pC_start + i ;                                 \
            GB_PUTC (cij, pC) ;                                         \
            Cb [pC] = 1 ;                                               \
            cnvals++ ;                                                  \
            break ;                                                     \
        }

    #endif

#else

    //--------------------------------------------------------------------------
    // all other semirings
    //--------------------------------------------------------------------------

    #define GB_DOT(k,pA,pB)                                             \
    {                                                                   \
        GB_GETA (aki, Ax, pA) ;  /* aki = A(k,i) */                     \
        GB_GETB (bkj, Bx, pB) ;  /* bkj = B(k,j) */                     \
        if (cij_exists)                                                 \
        {                                                               \
            /* cij += (A')(i,k) * B(k,j) */                             \
            GB_MULTADD (cij, aki, bkj, i, k, j) ;                       \
        }                                                               \
        else                                                            \
        {                                                               \
            /* cij = (A')(i,k) * B(k,j), and add to the pattern */      \
            cij_exists = true ;                                         \
            GB_MULT (cij, aki, bkj, i, k, j) ;                          \
        }                                                               \
        /* if (cij is terminal) break ; */                              \
        GB_DOT_TERMINAL (cij) ;                                         \
    }

#endif

//------------------------------------------------------------------------------
// C(i,j) = A(:,i)'*B(:,j): a single dot product
//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get the start of A(:,i) and B(:,j)
    //--------------------------------------------------------------------------

    int64_t pB = pB_start ;
    int64_t ainz = pA_end - pA ;
    ASSERT (ainz >= 0) ;
    bool cij_exists = false ;

    //--------------------------------------------------------------------------
    // declare the cij scalar
    //--------------------------------------------------------------------------

    GB_CIJ_DECLARE (cij) ;

    //--------------------------------------------------------------------------
    // 11 cases for computing C(i,j) = A(:,i)' * B(j,:)
    //--------------------------------------------------------------------------

#if 0
    if (A_is_bitmap && B_is_bitmap)
    {

        //----------------------------------------------------------------------
        // both A and B are bitmap
        //----------------------------------------------------------------------

        for (int64_t k = 0 ; k < vlen ; k++)
        { 
            if (!Ab [pA+k]) continue ;
            if (!Bb [pB+k]) continue ;
            GB_DOT (k, pA+k, pB+k) ;
        }

    }
    else if (A_is_bitmap)
    {

        //----------------------------------------------------------------------
        // A is bitmap; B is sparse, hypersparse, or full
        //----------------------------------------------------------------------

        for (int64_t p = pB ; p < pB_end ; p++)
        { 
            int64_t k = GBI (Bi, p, vlen) ;
            if (!Ab [pA+k]) continue ;
            GB_DOT (k, pA+k, p) ;
        }

    }
    else if (B_is_bitmap)
    {
#endif
        //----------------------------------------------------------------------
        // B is bitmap; A is sparse, hypersparse, or full
        //----------------------------------------------------------------------

        for (int64_t p = pA ; p < pA_end ; p++)
        { 
            int64_t k = Ai [p] ; // GBI (Ai, p, vlen) ;
            if (!Bb [pB+k]) continue ;
            GB_DOT (k, p, pB+k) ;
        }

#if 0
    }
    else if (ainz == 0)
    { 

        //----------------------------------------------------------------------
        // A(:,i) is empty so C(i,j) cannot be present
        //----------------------------------------------------------------------

        ;

    }
    else if (GBI (Ai, pA_end-1, vlen) < ib_first
        || ib_last < GBI (Ai, pA, vlen))
    { 

        //----------------------------------------------------------------------
        // pattern of A(:,i) and B(:,j) do not overlap
        //----------------------------------------------------------------------

        ;

    }
    else if (bjnz == vlen && ainz == vlen)
    {

        //----------------------------------------------------------------------
        // both A(:,i) and B(:,j) are dense
        //----------------------------------------------------------------------

        #if GB_IS_PAIR_MULTIPLIER

            #if GB_IS_ANY_MONOID
            // ANY monoid: take the first entry found; this sets cij = 1
            GB_MULT (cij, ignore, ignore, 0, 0, 0) ;
            #elif GB_IS_EQ_MONOID
            // EQ_PAIR semiring: all entries are equal to 1
            cij = 1 ;
            #elif (GB_CTYPE_BITS > 0)
            // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
            // for bool, 8-bit, 16-bit, or 32-bit integer
            cij = (GB_CTYPE) (((uint64_t) vlen) & GB_CTYPE_BITS) ;
            #else
            // PLUS monoid for float, double, or 64-bit integers 
            cij = GB_CTYPE_CAST (vlen, 0) ;
            #endif

        #else

            // cij = A(0,i) * B(0,j)
            GB_GETA (aki, Ax, pA) ;             // aki = A(0,i)
            GB_GETB (bkj, Bx, pB) ;             // bkj = B(0,j)
            GB_MULT (cij, aki, bkj, i, 0, j) ;  // cij = aki * bkj
            GB_PRAGMA_SIMD_DOT (cij)
            for (int64_t k = 1 ; k < vlen ; k++)
            { 
                GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                // cij += A(k,i) * B(k,j)
                GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
                GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
                GB_MULTADD (cij, aki, bkj, i, k, j) ; // cij += aki * bkj
            }

        #endif

        #if GB_CIJ_CHECK
        cij_exists = true ;
        #endif

    }
    else if (ainz == vlen)
    {

        //----------------------------------------------------------------------
        // A(:,i) is dense and B(:,j) is sparse
        //----------------------------------------------------------------------

        #if GB_IS_PAIR_MULTIPLIER

            #if GB_IS_ANY_MONOID
            // ANY monoid: take the first entry found; this sets cij = 1
            GB_MULT (cij, ignore, ignore, 0, 0, 0) ;
            #elif GB_IS_EQ_MONOID
            // EQ_PAIR semiring: all entries are equal to 1
            cij = 1 ;
            #elif (GB_CTYPE_BITS > 0)
            // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
            // for bool, 8-bit, 16-bit, or 32-bit integer
            cij = (GB_CTYPE) (((uint64_t) bjnz) & GB_CTYPE_BITS) ;
            #else
            // PLUS monoid for float, double, or 64-bit integers 
            cij = GB_CTYPE_CAST (bjnz, 0) ;
            #endif

        #else

            // first row index of B(:,j)
            int64_t k = Bi [pB] ;               // ok: B is sparse
            // cij = A(k,i) * B(k,j)
            GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
            GB_GETB (bkj, Bx, pB  ) ;           // bkj = B(k,j)
            GB_MULT (cij, aki, bkj, i, k, j) ;  // cij = aki * bkj
            GB_PRAGMA_SIMD_DOT (cij)
            for (int64_t p = pB+1 ; p < pB_end ; p++)
            { 
                GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                // next index of B(:,j)
                int64_t k = Bi [p] ;                // ok: B is sparse
                // cij += A(k,i) * B(k,j)
                GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
                GB_GETB (bkj, Bx, p   ) ;           // bkj = B(k,j)
                GB_MULTADD (cij, aki, bkj, i, k, j) ;   // cij += aki * bkj
            }

        #endif

        #if GB_CIJ_CHECK
        cij_exists = true ;
        #endif

    }
    else if (bjnz == vlen)
    {

        //----------------------------------------------------------------------
        // A(:,i) is sparse and B(:,j) is dense
        //----------------------------------------------------------------------

        #if GB_IS_PAIR_MULTIPLIER

            #if GB_IS_ANY_MONOID
            // ANY monoid: take the first entry found; this sets cij = 1
            GB_MULT (cij, ignore, ignore, 0, 0, 0) ;
            #elif GB_IS_EQ_MONOID
            // EQ_PAIR semiring: all entries are equal to 1
            cij = 1 ;
            #elif (GB_CTYPE_BITS > 0)
            // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
            // for bool, 8-bit, 16-bit, or 32-bit integer
            cij = (GB_CTYPE) (((uint64_t) ainz) & GB_CTYPE_BITS) ;
            #else
            // PLUS monoid for float, double, or 64-bit integers 
            cij = GB_CTYPE_CAST (ainz, 0) ;
            #endif

        #else

            // first row index of A(:,i)
            int64_t k = Ai [pA] ;               // ok: A is sparse
            // cij = A(k,i) * B(k,j)
            GB_GETA (aki, Ax, pA  ) ;           // aki = A(k,i)
            GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
            GB_MULT (cij, aki, bkj, i, k, j) ;  // cij = aki * bkj
            GB_PRAGMA_SIMD_DOT (cij)
            for (int64_t p = pA+1 ; p < pA_end ; p++)
            { 
                GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                // next index of A(:,i)
                int64_t k = Ai [p] ;                // ok: A is sparse
                // cij += A(k,i) * B(k,j)
                GB_GETA (aki, Ax, p   ) ;           // aki = A(k,i)
                GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
                GB_MULTADD (cij, aki, bkj, i, k, j) ;   // cij += aki * bkj
            }

        #endif

        #if GB_CIJ_CHECK
        cij_exists = true ;
        #endif

    }
    else if (ainz > 8 * bjnz)
    {

        //----------------------------------------------------------------------
        // B(:,j) is very sparse compared to A(:,i)
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {
            int64_t ia = Ai [pA] ;              // ok: A is sparse
            int64_t ib = Bi [pB] ;              // ok: B is sparse
            if (ia < ib)
            { 
                // A(ia,i) appears before B(ib,j)
                // discard all entries A(ia:ib-1,i)
                int64_t pleft = pA + 1 ;
                int64_t pright = pA_end - 1 ;
                GB_TRIM_BINARY_SEARCH (ib, Ai, pleft, pright) ;
                ASSERT (pleft > pA) ;
                pA = pleft ;
            }
            else if (ib < ia)
            { 
                // B(ib,j) appears before A(ia,i)
                pB++ ;
            }
            else // ia == ib == k
            { 
                // A(k,i) and B(k,j) are the next entries to merge
                GB_DOT (ia, pA, pB) ;
                pA++ ;
                pB++ ;
            }
        }

    }
    else if (bjnz > 8 * ainz)
    {

        //----------------------------------------------------------------------
        // A(:,i) is very sparse compared to B(:,j)
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {
            int64_t ia = Ai [pA] ;              // ok: A is sparse
            int64_t ib = Bi [pB] ;              // ok: B is sparse
            if (ia < ib)
            { 
                // A(ia,i) appears before B(ib,j)
                pA++ ;
            }
            else if (ib < ia)
            { 
                // B(ib,j) appears before A(ia,i)
                // discard all entries B(ib:ia-1,j)
                int64_t pleft = pB + 1 ;
                int64_t pright = pB_end - 1 ;
                GB_TRIM_BINARY_SEARCH (ia, Bi, pleft, pright) ;
                ASSERT (pleft > pB) ;
                pB = pleft ;
            }
            else // ia == ib == k
            { 
                // A(k,i) and B(k,j) are the next entries to merge
                GB_DOT (ia, pA, pB) ;
                pA++ ;
                pB++ ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A(:,i) and B(:,j) have about the same sparsity
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {
            int64_t ia = Ai [pA] ;              // ok: A is sparse
            int64_t ib = Bi [pB] ;              // ok: B is sparse
            if (ia == ib)
            { 
                GB_DOT (ia, pA, pB) ;
                pA++ ;
                pB++ ;
            }
            else
            { 
                pA += (ia < ib) ;
                pB += (ib < ia) ;
            }
        }
    }
#endif

    //--------------------------------------------------------------------------
    // save C(i,j)
    //--------------------------------------------------------------------------

    #if defined ( GB_DOT3 )

        // GB_AxB_dot3: computing C<M>=A'*B; C and M are sparse/hypersparse
        if (GB_CIJ_EXISTS)
        { 
            // C(i,j) = cij
            GB_PUTC (cij, pC) ;         // Cx [pC] = cij
            Ci [pC] = i ;               // ok: C is sparse
        }
        else
        { 
            // C(i,j) becomes a zombie
            task_nzombies++ ;           // GB_AxB_dot3: computing C<M>=A'*B
            Ci [pC] = GB_FLIP (i) ;     // ok: C is sparse
        }

    #else

        // GB_AxB_dot2: computing C=A'*B, C<M>=A'*B, or C<!M>=A'*B, where M, A,
        // B, can have any sparsity pattern, and C is bitmap.  The ANY monoid
        // writes its values to C immediately.

        #if ( !GB_IS_ANY_MONOID )
        if (GB_CIJ_EXISTS)
        { 
            int64_t pC = pC_start + i ;
            GB_PUTC (cij, pC) ;
            Cb [pC] = 1 ;
            cnvals++ ;
        }
        #endif

    #endif
}

#undef GB_DOT
#undef GB_CIJ_EXISTS
#undef GB_CIJ_CHECK

                }
            }
        }
    }

    C->nvals = cnvals ;
}

#undef GB_MASK_IS_PRESENT

    }
}



    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<M>=A'*B: masked dot product method (phase 2)
//------------------------------------------------------------------------------

GrB_Info GB_Adot3B__any_firstj_int32
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    const GB_task_struct *GB_RESTRICT TaskList,
    const int ntasks,
    const int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_AxB_dot3_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C+=A'*B: dense dot product
//------------------------------------------------------------------------------

GrB_Info GB_Adot4B__any_firstj_int32
(
    GrB_Matrix C,
    const GrB_Matrix A, bool A_is_pattern,
    int64_t *GB_RESTRICT A_slice, int naslice,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *GB_RESTRICT B_slice, int nbslice,
    const int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_AxB_dot4_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C=A*B, C<M>=A*B, C<!M>=A*B: saxpy3 method (Gustavson + Hash)
//------------------------------------------------------------------------------

#include "GB_AxB_saxpy3_template.h"

GrB_Info GB_Asaxpy3B__any_firstj_int32
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const bool M_dense_in_place,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    GB_saxpy3task_struct *GB_RESTRICT TaskList,
    int ntasks,
    int nfine,
    int nthreads,
    GB_Context Context
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_AxB_saxpy_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

#endif

