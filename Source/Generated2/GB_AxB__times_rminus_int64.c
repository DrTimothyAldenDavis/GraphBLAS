
//------------------------------------------------------------------------------
// GB_AxB__times_rminus_int64.c: matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated1/ or Generated2/ folder, do not edit it
// (it is auto-generated from Generator/*).

#include "GB_dev.h"

#ifndef GBCUDA_DEV

#include "GB.h"
#include "GB_control.h"
#include "GB_sort.h"
#include "GB_AxB_saxpy.h"
#if 1
#include "GB_AxB__include2.h"
#else
#include "GB_AxB__include1.h"
#endif
#include "GB_unused.h"
#include "GB_bitmap_assign_methods.h"
#include "GB_ek_slice_search.c"

// This C=A*B semiring is defined by the following types and operators:

// A'*B (dot2):        GB (_Adot2B__times_rminus_int64)
// A'*B (dot3):        GB (_Adot3B__times_rminus_int64)
// C+=A'*B (dot4):     GB (_Adot4B__times_rminus_int64)
// A*B (saxpy bitmap): GB (_AsaxbitB__times_rminus_int64)
// A*B (saxpy3):       GB (_Asaxpy3B__times_rminus_int64)
//     no mask:        GB (_Asaxpy3B_noM__times_rminus_int64)
//     mask M:         GB (_Asaxpy3B_M__times_rminus_int64)
//     mask !M:        GB (_Asaxpy3B_notM__times_rminus_int64)
// A*B (saxpy4):       GB (_Asaxpy4B__times_rminus_int64)
// A*B (saxpy5):       GB (_Asaxpy5B__times_rminus_int64)

// C type:     int64_t
// A type:     int64_t
// A pattern?  0
// B type:     int64_t
// B pattern?  0

// Multiply: z = (y - x)
// Add:      cij *= t
//    'any' monoid?  0
//    atomic?        1
//    OpenMP atomic? 1
//    identity:      1
//    terminal?      1
//    terminal:      if (z == 0) { break ; }
// MultAdd:  { int64_t x_op_y = (y - x) ; z *= x_op_y ; }

#define GB_A_TYPE \
    int64_t

#define GB_B_TYPE \
    int64_t

#define GB_C_TYPE \
    int64_t

#define GB_A_ISO A_iso
#define GB_B_ISO B_iso
#define GB_C_ISO \
    0

// true for int64, uint64, float, double, float complex, and double complex 
#define GB_ZTYPE_IGNORE_OVERFLOW \
    1

// declare aik as atype
#define GB_DECLAREA(aik) \
    int64_t aik

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA,A_iso) \
    aik = GBX (Ax, pA, A_iso)

// true if values of A are not used
#define GB_A_IS_PATTERN \
    0 \

// declare bkj as btype
#define GB_DECLAREB(bkj) \
    int64_t bkj

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB,B_iso) \
    bkj = GBX (Bx, pB, B_iso)

// true if values of B are not used
#define GB_B_IS_PATTERN \
    0 \

// multiply operator
#define GB_MULT(z, x, y, i, k, j) \
    z = (y - x)

// FIXME: GB_CTYPE_CAST not in macrofy (for PLUS_PAIR, and t=1 for PAIR operator)
// cast from a real scalar (or 2, if C is complex) to the type of C
// Should be to ztype
#define GB_CTYPE_CAST(x,y) \
    ((int64_t) x)

// multiply-add
#define GB_MULTADD(z, x, y, i, k, j) \
    { int64_t x_op_y = (y - x) ; z *= x_op_y ; }

// FIXME: GB_IDENTITY only appears in a few templates; replace it
// monoid identity value
#define GB_IDENTITY \
    1

// declare and initialize z = identity value of the monoid
#define GB_DECLARE_MONOID_IDENTITY(z) \
    int64_t z = 1 ;

// FIXME: GB_HAS_IDENTITY_BYTE not in macrofy (add it)
// 1 if the identity value can be assigned via memset, with all bytes the same
#define GB_HAS_IDENTITY_BYTE \
    0

// FIXME: GB_IDENTITY_BYTE not in macrofy (add it)
// identity byte, for memset
#define GB_IDENTITY_BYTE \
    (none)

// true if the monoid has a terminal value
#define GB_MONOID_IS_TERMINAL \
    1

// break if z reaches the terminal value (dot product only)
#define GB_IF_TERMINAL_BREAK(z,zterminal) \
    if (z == 0) { break ; }

// FIXME: GB_PRAGMA_SIMD_DOT not in macrofy, do I need it?
// simd pragma for dot-product loop vectorization
#define GB_PRAGMA_SIMD_DOT(cij) \
    ;

// FIXME: GB_PRAGMA_SIMD_VECTORIZE: move this (generic methods disable it)
// simd pragma for other loop vectorization
#define GB_PRAGMA_SIMD_VECTORIZE GB_PRAGMA_SIMD

// 1 for the PLUS_PAIR_(real) semirings, not for the complex case
#define GB_IS_PLUS_PAIR_REAL_SEMIRING \
    0

// FIXME: GB_CIJ_DECLARE(cij) do I need this?  Use GB_DECLARE_MONOID_IDENTITY(cij) instead?
// declare the cij scalar (initialize cij to zero for PLUS_PAIR)
#define GB_CIJ_DECLARE(cij) \
    int64_t cij

// Cx [pC] = cij
#define GB_PUTC(cij,p) \
    Cx [p] = cij

// FIXME: GB_CIJ_WRITE(p,t) why do I need this?  typecast?
// change to use GB_PUTC instead
// Cx [p] = t
#define GB_CIJ_WRITE(p,t) \
    Cx [p] = t

// FIXME: GB_CIJ_UPDATE(p,t) why do I need this?  Use GB_UPDATE instead?
// C(i,j) += t
#define GB_CIJ_UPDATE(p,t) \
    Cx [p] *= t

// z = x + y
#define GB_ADD(z,x,y) \
    z = x * y

// FIXME: GB_CTYPE_BITS for PLUS_PAIR semirings only, 0 otherwise
// bit pattern for bool, 8-bit, 16-bit, and 32-bit integers
#define GB_CTYPE_BITS \
    0

// 1 if monoid update can skipped entirely (the ANY monoid)
#define GB_IS_ANY_MONOID \
    0

// FIXME: GB_IS_EQ_MONOID
// 1 if monoid update is EQ
#define GB_IS_EQ_MONOID \
    0

// FIXME: GB_HAS_ATOMIC
// 1 if monoid update can be done atomically, 0 otherwise
#define GB_HAS_ATOMIC \
    1

// FIXME: GB_HAS_OMP_ATOMIC (general and MSVC)
// 1 if monoid update can be done with an OpenMP atomic update, 0 otherwise
#if GB_COMPILER_MSC
    /* MS Visual Studio only has OpenMP 2.0, with fewer atomics */
    #define GB_HAS_OMP_ATOMIC \
        1
#else
    #define GB_HAS_OMP_ATOMIC \
        1
#endif

// FIXME: GB_IS_ANY_PAIR_SEMIRING
// 1 for the ANY_PAIR_ISO semiring
#define GB_IS_ANY_PAIR_SEMIRING \
    0

// FIXME: GB_IS_PAIR_MULTIPLIER
// 1 if PAIR is the multiply operator 
#define GB_IS_PAIR_MULTIPLIER \
    0

// FIXME: GB_IS_PLUS_FC32_MONOID
// 1 if monoid is PLUS_FC32
#define GB_IS_PLUS_FC32_MONOID \
    0

// FIXME: GB_IS_PLUS_FC64_MONOID
// 1 if monoid is PLUS_FC64
#define GB_IS_PLUS_FC64_MONOID \
    0

// FIXME: GB_IS_ANY_FC32_MONOID
// 1 if monoid is ANY_FC32
#define GB_IS_ANY_FC32_MONOID \
    0

// FIXME: GB_IS_ANY_FC64_MONOID
// 1 if monoid is ANY_FC64
#define GB_IS_ANY_FC64_MONOID \
    0

// FIXME: GB_IS_IMIN_MONOID 
// 1 if monoid is MIN for signed or unsigned integers
#define GB_IS_IMIN_MONOID \
    0

// FIXME: GB_IS_IMAX_MONOID 
// 1 if monoid is MAX for signed or unsigned integers
#define GB_IS_IMAX_MONOID \
    0

// FIXME: GB_IS_FMIN_MONOID 
// 1 if monoid is MIN for float or double
#define GB_IS_FMIN_MONOID \
    0

// FIXME: GB_IS_FMAX_MONOID 
// 1 if monoid is MAX for float or double
#define GB_IS_FMAX_MONOID \
    0

// FIXME: GB_IS_FMIN_MONOID 
// 1 for the FIRSTI or FIRSTI1 multiply operator
#define GB_IS_FIRSTI_MULTIPLIER \
    0

// FIXME: GB_IS_FIRSTJ_MONOID  or FIRSTJ1
// 1 for the FIRSTJ or FIRSTJ1 multiply operator
#define GB_IS_FIRSTJ_MULTIPLIER \
    0

// FIXME: GB_IS_SECONDJ_MONOID  or SECONDJ1
// 1 for the SECONDJ or SECONDJ1 multiply operator
#define GB_IS_SECONDJ_MULTIPLIER \
    0

// FIXME: GB_OFFSET for FIRST[IJ]1 and SECOND[IJ]1
// 1 for the FIRSTI1, FIRSTJ1, SECONDI1, or SECONDJ1 multiply operators
#define GB_OFFSET \
    0

// FIXME: GB_ATOMIC_COMPARE_EXCHANGE
// atomic compare-exchange
#define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
    GB_ATOMIC_COMPARE_EXCHANGE_64 (target, expected, desired)

// FIXME: GB_HX_WRITE
// Hx [i] = t
#define GB_HX_WRITE(i,t) \
    Hx [i] = t

// FIXME: GB_CIJ_GATHER
// Cx [p] = Hx [i]
#define GB_CIJ_GATHER(p,i) \
    Cx [p] = Hx [i]

// FIXME: GB_CIJ_GATHER_UPDATE(p,i)
// Cx [p] += Hx [i]
#define GB_CIJ_GATHER_UPDATE(p,i) \
    Cx [p] *= Hx [i]

// FIXME: GB_HX_UPDATE(i,t)
// Hx [i] += t
#define GB_HX_UPDATE(i,t) \
    Hx [i] *= t

// FIXME: GB_CIJ_MEMCPY(p,i,len)
// memcpy (&(Cx [p]), &(Hx [i]), len)
#define GB_CIJ_MEMCPY(p,i,len) \
    memcpy (Cx +(p), Hx +(i), (len) * sizeof(int64_t));

// disable this semiring and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_TIMES || GxB_NO_RMINUS || GxB_NO_INT64 || GxB_NO_TIMES_INT64 || GxB_NO_RMINUS_INT64 || GxB_NO_TIMES_RMINUS_INT64)

//------------------------------------------------------------------------------
// GB_Adot2B: C=A'*B, C<M>=A'*B, or C<!M>=A'*B: dot product method, C is bitmap
//------------------------------------------------------------------------------

// if A_not_transposed is true, then C=A*B is computed where A is bitmap or full

GrB_Info GB (_Adot2B__times_rminus_int64)
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const bool A_not_transposed,
    const GrB_Matrix A, int64_t *restrict A_slice,
    const GrB_Matrix B, int64_t *restrict B_slice,
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
// GB_Adot3B: C<M>=A'*B: masked dot product, C is sparse or hyper
//------------------------------------------------------------------------------

GrB_Info GB (_Adot3B__times_rminus_int64)
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_task_struct *restrict TaskList,
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
// GB_Adot4B:  C+=A'*B: dense dot product
//------------------------------------------------------------------------------

    GrB_Info GB (_Adot4B__times_rminus_int64)
    (
        GrB_Matrix C,
        const GrB_Matrix A, int64_t *restrict A_slice, int naslice,
        const GrB_Matrix B, int64_t *restrict B_slice, int nbslice,
        const int nthreads,
        GB_Werk Werk
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
// GB_AsaxbitB: C=A*B, C<M>=A*B, C<!M>=A*B: saxpy method, C is bitmap/full
//------------------------------------------------------------------------------

#include "GB_AxB_saxpy3_template.h"

GrB_Info GB (_AsaxbitB__times_rminus_int64)
(
    GrB_Matrix C,   // bitmap or full
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_bitmap_AxB_saxpy_template.c"
    return (GrB_SUCCESS) ;
    #endif
}


//------------------------------------------------------------------------------
// GB_Asaxpy4B: C += A*B when C is full
//------------------------------------------------------------------------------

    GrB_Info GB (_Asaxpy4B__times_rminus_int64)
    (
        GrB_Matrix C,
        const GrB_Matrix A,
        const GrB_Matrix B,
        const int ntasks,
        const int nthreads,
        const int nfine_tasks_per_vector,
        const bool use_coarse_tasks,
        const bool use_atomics,
        const int64_t *A_slice,
        GB_Werk Werk
    )
    { 
        #if GB_DISABLE
        return (GrB_NO_VALUE) ;
        #else
        #include "GB_AxB_saxpy4_template.c"
        return (GrB_SUCCESS) ;
        #endif
    }



//------------------------------------------------------------------------------
// GB_Asaxpy5B: C += A*B when C is full, A is bitmap/full, B is sparse/hyper
//------------------------------------------------------------------------------

    #if GB_DISABLE
    #elif ( !GB_A_IS_PATTERN )

        

        //----------------------------------------------------------------------
        // saxpy5 method unrolled, with no vectors
        //----------------------------------------------------------------------

        #undef  GB_V16
        #undef  GB_V8
        #undef  GB_V4

        #define GB_V16 0
        #define GB_V8  0
        #define GB_V4  0

        static inline void GB_AxB_saxpy5_unrolled_vanilla
        (
            GrB_Matrix C,
            const GrB_Matrix A,
            const GrB_Matrix B,
            const int ntasks,
            const int nthreads,
            const int64_t *B_slice,
            GB_Werk Werk
        )
        {
            #include "GB_AxB_saxpy5_unrolled.c"
        }

    #endif

    GrB_Info GB (_Asaxpy5B__times_rminus_int64)
    (
        GrB_Matrix C,
        const GrB_Matrix A,
        const GrB_Matrix B,
        const int ntasks,
        const int nthreads,
        const int64_t *B_slice,
        GB_Werk Werk
    )
    { 
        #if GB_DISABLE
        return (GrB_NO_VALUE) ;
        #else
        #include "GB_AxB_saxpy5_meta.c"
        return (GrB_SUCCESS) ;
        #endif
    }


//------------------------------------------------------------------------------
// GB_Asaxpy3B: C=A*B, C<M>=A*B, C<!M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

GrB_Info GB (_Asaxpy3B__times_rminus_int64)
(
    GrB_Matrix C,   // C<any M>=A*B, C sparse or hypersparse
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks, const int nfine, const int nthreads, const int do_sort,
    GB_Werk Werk
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
    if (M == NULL)
    {
        // C = A*B, no mask
        return (GB (_Asaxpy3B_noM__times_rminus_int64) (C, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    else if (!Mask_comp)
    {
        // C<M> = A*B
        return (GB (_Asaxpy3B_M__times_rminus_int64) (C,
            M, Mask_struct, M_in_place, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    else
    {
        // C<!M> = A*B
        return (GB (_Asaxpy3B_notM__times_rminus_int64) (C,
            M, Mask_struct, M_in_place, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_Asaxpy3B_M: C<M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#if ( !GB_DISABLE )

    GrB_Info GB (_Asaxpy3B_M__times_rminus_int64)
    (
        GrB_Matrix C,   // C<M>=A*B, C sparse or hypersparse
        const GrB_Matrix M, const bool Mask_struct,
        const bool M_in_place,
        const GrB_Matrix A,
        const GrB_Matrix B,
        GB_saxpy3task_struct *restrict SaxpyTasks,
        const int ntasks, const int nfine, const int nthreads,
        const int do_sort,
        GB_Werk Werk
    )
    {
        if (GB_IS_SPARSE (A) && GB_IS_SPARSE (B))
        {
            // both A and B are sparse
            #define GB_META16
            #define GB_NO_MASK 0
            #define GB_MASK_COMP 0
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_definitions.h"
            #include "GB_AxB_saxpy3_template.c"
        }
        else
        {
            // general case
            #undef GB_META16
            #define GB_NO_MASK 0
            #define GB_MASK_COMP 0
            #include "GB_meta16_definitions.h"
            #include "GB_AxB_saxpy3_template.c"
        }
        return (GrB_SUCCESS) ;
    }

#endif

//------------------------------------------------------------------------------
//GB_Asaxpy3B_noM: C=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#if ( !GB_DISABLE )

    GrB_Info GB (_Asaxpy3B_noM__times_rminus_int64)
    (
        GrB_Matrix C,   // C=A*B, C sparse or hypersparse
        const GrB_Matrix A,
        const GrB_Matrix B,
        GB_saxpy3task_struct *restrict SaxpyTasks,
        const int ntasks, const int nfine, const int nthreads,
        const int do_sort,
        GB_Werk Werk
    )
    {
        if (GB_IS_SPARSE (A) && GB_IS_SPARSE (B))
        {
            // both A and B are sparse
            #define GB_META16
            #define GB_NO_MASK 1
            #define GB_MASK_COMP 0
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_definitions.h"
            #include "GB_AxB_saxpy3_template.c"
        }
        else
        {
            // general case
            #undef GB_META16
            #define GB_NO_MASK 1
            #define GB_MASK_COMP 0
            #include "GB_meta16_definitions.h"
            #include "GB_AxB_saxpy3_template.c"
        }
        return (GrB_SUCCESS) ;
    }

#endif

//------------------------------------------------------------------------------
//GB_Asaxpy3B_notM: C<!M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#if ( !GB_DISABLE )

    GrB_Info GB (_Asaxpy3B_notM__times_rminus_int64)
    (
        GrB_Matrix C,   // C<!M>=A*B, C sparse or hypersparse
        const GrB_Matrix M, const bool Mask_struct,
        const bool M_in_place,
        const GrB_Matrix A,
        const GrB_Matrix B,
        GB_saxpy3task_struct *restrict SaxpyTasks,
        const int ntasks, const int nfine, const int nthreads,
        const int do_sort,
        GB_Werk Werk
    )
    {
        if (GB_IS_SPARSE (A) && GB_IS_SPARSE (B))
        {
            // both A and B are sparse
            #define GB_META16
            #define GB_NO_MASK 0
            #define GB_MASK_COMP 1
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_definitions.h"
            #include "GB_AxB_saxpy3_template.c"
        }
        else
        {
            // general case
            #undef GB_META16
            #define GB_NO_MASK 0
            #define GB_MASK_COMP 1
            #include "GB_meta16_definitions.h"
            #include "GB_AxB_saxpy3_template.c"
        }
        return (GrB_SUCCESS) ;
    }

#endif
#endif

