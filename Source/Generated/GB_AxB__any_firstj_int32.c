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
//     #include "GB_AxB_saxpy_template.c"

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

