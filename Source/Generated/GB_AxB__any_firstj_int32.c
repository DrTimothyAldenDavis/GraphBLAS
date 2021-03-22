//------------------------------------------------------------------------------
// GB_AxB:  hard-coded functions for semiring: C<M>=A*B or A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated).

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "GB_bracket.h"
#include "GB_sort.h"
#include "GB_atomics.h"
#include "GB_AxB_saxpy.h"
#include "GB_AxB__include.h"
#include "GB_unused.h"
#include "GB_bitmap_assign_methods.h"
#include "GB_ek_slice_search.c"

// The C=A*B semiring is defined by the following types and operators:

// A'*B function (dot2):     GB_Adot2B__any_firstj_int32
// A'*B function (dot3):     GB_Adot3B__any_firstj_int32
// C+=A'*B function (dot4):  GB_Adot4B__any_firstj_int32
// A*B function (saxpy):     GB_AsaxpyB__any_firstj_int32

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

#define GB_ASIZE (sizeof (GB_BTYPE))
#define GB_BSIZE (sizeof (GB_BTYPE))
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

// Gx [pG] = Ax [pA]
#define GB_LOADA(Gx,pG,Ax,pA) \
    Gx [pG] = Ax [pA]

// Gx [pG] = Bx [pB]
#define GB_LOADB(Gx,pG,Bx,pB) \
    ;

#define GB_CX(p) Cx [p]

// multiply operator
#define GB_MULT(z, x, y, i, k, j) \
    z = k

// cast from a real scalar (or 2, if C is complex) to the type of C
#define GB_CTYPE_CAST(x,y) \
    ((int32_t) x)

// cast from a real scalar (or 2, if A is complex) to the type of A
#define GB_ATYPE_CAST(x,y) \
    ((int32_t) x)

// multiply-add
#define GB_MULTADD(z, x, y, i, k, j) \
    z = k

// monoid identity value
#define GB_IDENTITY \
    0

// 1 if the identity value can be assigned via memset, with all bytes the same
#define GB_HAS_IDENTITY_BYTE \
    0

// identity byte, for memset
#define GB_IDENTITY_BYTE \
    (none)

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

// 1 for performance-critical semirings, which get extra optimization
#define GB_IS_PERFORMANCE_CRITICAL_SEMIRING \
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

// 1 if monoid is ANY_FC32
#define GB_IS_ANY_FC32_MONOID \
    0

// 1 if monoid is ANY_FC64
#define GB_IS_ANY_FC64_MONOID \
    0

// 1 if monoid is MIN for signed or unsigned integers
#define GB_IS_IMIN_MONOID \
    0

// 1 if monoid is MAX for signed or unsigned integers
#define GB_IS_IMAX_MONOID \
    0

// 1 if monoid is MIN for float or double
#define GB_IS_FMIN_MONOID \
    0

// 1 if monoid is MAX for float or double
#define GB_IS_FMAX_MONOID \
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
    #define GB_CIJ_GATHER_UPDATE(p,i)
    #define GB_HX_UPDATE(i,t)
    #define GB_CIJ_MEMCPY(p,i,len)

#else

    // Hx [i] = t
    #define GB_HX_WRITE(i,t) Hx [i] = t

    // Hx [i] = identity
    #define GB_HX_CLEAR(i) Hx [i] = GB_IDENTITY

    // Cx [p] = Hx [i]
    #define GB_CIJ_GATHER(p,i) Cx [p] = Hx [i]

    // Cx [p] += Hx [i]
    #define GB_CIJ_GATHER_UPDATE(p,i) \
        Cx [p] = Hx [i]

    // Hx [i] += t
    #define GB_HX_UPDATE(i,t) \
        Hx [i] = t

    // memcpy (&(Cx [p]), &(Hx [i]), len)
    #define GB_CIJ_MEMCPY(p,i,len) \
        memcpy (Cx +(p), Hx +(i), (len) * sizeof(int32_t))

#endif

// 1 if the semiring has a concise bitmap multiply-add
#define GB_HAS_BITMAP_MULTADD \
    1

// concise statement(s) for the bitmap case:
//  if (exists)
//      if (cb == 0)
//          cx = ax * bx
//          cb = 1
//      else
//          cx += ax * bx
#define GB_BITMAP_MULTADD(cb,cx,exists,ax,bx) \
    if (exists && !cb) cx = (k) ; cb |= exists

// define X for bitmap multiply-add
#define GB_XINIT \
    ;

// load X [1] = bkj for bitmap multiply-add
#define GB_XLOAD(bkj) \
    ;

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
    #include "GB_AxB_dot2_meta.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<M>=A'*B: masked dot product method (phase 2) where C is sparse or hyper
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
    #include "GB_AxB_dot3_meta.c"
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
    #include "GB_AxB_dot4_meta.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C=A*B, C<M>=A*B, C<!M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#include "GB_AxB_saxpy3_template.h"

GrB_Info GB_AsaxpyB__any_firstj_int32
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const bool M_dense_in_place,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    const int saxpy_method,
    // for saxpy3 method only:
    GB_saxpy3task_struct *GB_RESTRICT SaxpyTasks,
    int ntasks, int nfine,
    // for saxpy3 and saxpy4 methods only:
    int nthreads,
    const int do_sort,
    // for saxpy4 method only:
    int8_t  *GB_RESTRICT Wf,
    int64_t **Wi_handle,
    size_t Wi_size,
    GB_void *GB_RESTRICT Wx,
    int64_t *GB_RESTRICT kfirst_Bslice,
    int64_t *GB_RESTRICT klast_Bslice,
    int64_t *GB_RESTRICT pstart_Bslice,
    GB_Context Context
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    // #include "GB_AxB_saxpy_template.c"
//------------------------------------------------------------------------------
// GB_AxB_saxpy_template: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All 4 matrices have any format: hypersparse, sparse, bitmap, or full.

{
    switch (saxpy_method)
    {

        case GB_SAXPY_METHOD_3 :
        { 
            // C is sparse or hypersparse, using minimal workspace.
            ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
            #include "GB_AxB_saxpy3_template.c"
        }
        break ;

        case GB_SAXPY_METHOD_4 :
        { 
            // C is sparse or hypersparse, using large workspace
            ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
            ASSERT (GB_IS_SPARSE (A)) ;
            ASSERT (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B)) ;
            // #include "GB_AxB_saxpy4_template.c"
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

    const int32_t *GB_RESTRICT Mxi = (int32_t *) Mx ;

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
                int8_t mij = Mb [pM+i] ; \
                /* bool mij = GBB (Mb, pM+i) && GB_mcast (Mx, pM+i, msize) ;  */ \
                if (mij) continue ;
            #undef  GB_CHECK_BITMAP_OR_FULL_MASK
            #define GB_CHECK_BITMAP_OR_FULL_MASK(i) GB_CHECK_MASK(i)
            // #include "GB_AxB_saxpy4_phase1.c"
//------------------------------------------------------------------------------
// GB_AxB_saxpy4_phase1: compute the pattern of C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{
    if (nthreads == 1)
    {
// printf ("saxpy4 phase1 , 1 thread\n") ;
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
            // get M(:,j) where j = Bh [kk], if M is bitmap/full
            GB_GET_MASK_j ;

            const int8_t *GB_RESTRICT Mbj = Mb + pM ;

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
                    int64_t i = Ai [pA] ;
                    /*
                    Mb [i] == 0: entry can be modified
                        Hf [i] == 0:    C(i,j) is a new entry, set Hf [i] to 2
                        Hf [i] == 2:    C(i,j) is an existing entry, keep Hf [i] as 2
                        1-mij = 1-0 = 1

                    Mb [i] == 1: entry cannot be modified
                        Hf [i] == 0:    must remain zero
                        1-mij = 1-1 = 0
                    */
                    if ((Mbj [i] == 0) && (Hf [i] == 0))
                    {
                        Hf [i] = 2 ;
                        Wi [pC] = i ; 
                        pC ++ ;
                        Hx [i] = k ;
                    }

#if 0
                    // get A(i,k)
                    int64_t i = Ai [pA] ;
                    // check M(i,j) if M is bitmap/full
                    // GB_CHECK_BITMAP_OR_FULL_MASK (i) ;
                    if (Mbj [i]) continue ;
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
                    printf ("SORT!!\n") ;
                    GB_qsort_1a (Wi + pC_start, cknz) ;
                }
            }
        }

        //----------------------------------------------------------------------
        // log the end of the last vector of C
        //----------------------------------------------------------------------

        Cp [bnvec] = pC ;
//      printf ("cnz %ld\n", pC) ;

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
                // get M(:,j), if M is bitmap or full
                GB_GET_MASK_j ;

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

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (3, ttt) ;
ttt = omp_get_wtime ( ) ;

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

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (6, ttt) ;
ttt = omp_get_wtime ( ) ;
}

        }
        break ;

        case GB_SAXPY_METHOD_BITMAP :
        { 
            // C is bitmap or full
            #include "GB_bitmap_AxB_saxpy_template.c"
        }

        default:;
    }
}

    return (GrB_SUCCESS) ;
    #endif
}

#endif

