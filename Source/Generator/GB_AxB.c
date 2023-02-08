//------------------------------------------------------------------------------
// GB_AxB.c: matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated1/ or Generated2/ folder, do not edit it
// (it is auto-generated from Generator/*).

#include "GB_dev.h"

ifndef_GBCUDA_DEV

#include "GB.h"
#include "GB_control.h"
#include "GB_sort.h"
#include "GB_AxB_saxpy.h"
if_not_any_pair_semiring
#include "GB_AxB__include2.h"
#else
#include "GB_AxB__include1.h"
#endif
#include "GB_unused.h"
#include "GB_bitmap_assign_methods.h"
#include "GB_ek_slice_search.c"

// This C=A*B semiring is defined by the following types and operators:

// A'*B (dot2):        GB (_Adot2B)
// A'*B (dot3):        GB (_Adot3B)
// C+=A'*B (dot4):     GB (_Adot4B)
// A*B (saxpy bitmap): GB (_AsaxbitB)
// A*B (saxpy3):       GB (_Asaxpy3B)
//     no mask:        GB (_Asaxpy3B_noM)
//     mask M:         GB (_Asaxpy3B_M)
//     mask !M:        GB (_Asaxpy3B_notM)
// A*B (saxpy4):       GB (_Asaxpy4B)
// A*B (saxpy5):       GB (_Asaxpy5B)

// semiring operators:
GB_multiply_add
GB_multiply
GB_add_op
GB_add_update
// identity: GB_identity

// types: C, A, B matrix types; A and B cast to A2 and B2; Z is the monoid type
GB_atype
GB_a2type
GB_btype
GB_b2type
GB_ztype
GB_ctype

// iso and pattern cases:
#define GB_A_ISO A_iso
#define GB_B_ISO B_iso
GB_c_iso
GB_a_is_pattern
GB_b_is_pattern

// special case semirings:
GB_is_any_pair_semiring
GB_is_plus_pair_real_semiring
GB_is_eq_pair_semiring
GB_is_xor_pair_semiring
GB_is_plus_8_pair_semiring
GB_is_plus_16_pair_semiring
GB_is_plus_32_pair_semiring
GB_is_plus_big_pair_semiring
GB_is_plus_fc32_pair_semiring
GB_is_plus_fc64_pair_semiring

// special case monoids:
GB_is_any_fc32_monoid
GB_is_any_fc64_monoid
GB_is_imin_monoid
GB_is_imax_monoid
GB_is_fmin_monoid
GB_is_fmax_monoid
GB_is_plus_fc32_monoid
GB_is_plus_fc64_monoid

// special case multipliers:
GB_is_pair_multiplier
GB_pair_one
GB_offset
GB_is_firsti_multiplier
GB_is_firstj_multiplier
GB_is_secondj_multiplier

// z = identity, and ztype overflow condition (if any):
GB_declare_monoid_identity
GB_has_identity_byte
GB_identity_byte
GB_ztype_ignore_overflow

// monoid terminal condition, if any:
GB_is_any_monoid
GB_monoid_is_terminal
GB_terminal_condition
GB_if_terminal_break
GB_declare_monoid_terminal

// declare aik as a2type
#define GB_DECLAREA(aik) \
    GB_declarea(aik)

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA,A_iso) \
    GB_geta(aik,Ax,pA,A_iso)

// declare bkj as b2type
#define GB_DECLAREB(bkj) \
    GB_declareb(bkj)

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB,B_iso) \
    GB_getb(bkj,Bx,pB,B_iso)

// Cx [pC] = cij
#define GB_PUTC(cij,p) \
    GB_putc

// FIXME: GB_IDENTITY only appears in a few templates; replace it
// monoid identity value
#define GB_IDENTITY \
    GB_identity

// FIXME: GB_PRAGMA_SIMD_DOT not in macrofy, do I need it?
// simd pragma for dot-product loop vectorization
#define GB_PRAGMA_SIMD_DOT(cij) \
    GB_dot_simd_vectorize(cij)

// FIXME: GB_PRAGMA_SIMD_VECTORIZE: move this (generic methods disable it)
// simd pragma for other loop vectorization
#define GB_PRAGMA_SIMD_VECTORIZE GB_PRAGMA_SIMD

// FIXME: GB_HAS_ATOMIC
// 1 if monoid update can be done atomically, 0 otherwise
#define GB_HAS_ATOMIC \
    GB_has_atomic

// FIXME: GB_HAS_OMP_ATOMIC (general and MSVC)
// 1 if monoid update can be done with an OpenMP atomic update, 0 otherwise
#if GB_COMPILER_MSC
    /* MS Visual Studio only has OpenMP 2.0, with fewer atomics */
    #define GB_HAS_OMP_ATOMIC \
        GB_microsoft_has_omp_atomic
#else
    #define GB_HAS_OMP_ATOMIC \
        GB_has_omp_atomic
#endif

// FIXME: GB_ATOMIC_COMPARE_EXCHANGE
// atomic compare-exchange
#define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
    GB_atomic_compare_exchange

// disable this semiring and use the generic case if these conditions hold
#define GB_DISABLE \
    GB_disable

// finalize anything not yet defined
#include "GB_AxB_shared_definitions.h"

//------------------------------------------------------------------------------
// GB_Adot2B: C=A'*B, C<M>=A'*B, or C<!M>=A'*B: dot product method, C is bitmap
//------------------------------------------------------------------------------

// if A_not_transposed is true, then C=A*B is computed where A is bitmap or full

GrB_Info GB (_Adot2B)
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

GrB_Info GB (_Adot3B)
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

m4_divert(if_dot4_enabled)
//------------------------------------------------------------------------------
// GB_Adot4B:  C+=A'*B: dense dot product
//------------------------------------------------------------------------------

    GrB_Info GB (_Adot4B)
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
m4_divert(0)

//------------------------------------------------------------------------------
// GB_AsaxbitB: C=A*B, C<M>=A*B, C<!M>=A*B: saxpy method, C is bitmap only
//------------------------------------------------------------------------------

#include "GB_AxB_saxpy3_template.h"

GrB_Info GB (_AsaxbitB)
(
    GrB_Matrix C,   // bitmap only
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

m4_divert(if_saxpy4_enabled)
//------------------------------------------------------------------------------
// GB_Asaxpy4B: C += A*B when C is full
//------------------------------------------------------------------------------

    GrB_Info GB (_Asaxpy4B)
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
m4_divert(0)

m4_divert(if_saxpy5_enabled)
//------------------------------------------------------------------------------
// GB_Asaxpy5B: C += A*B when C is full, A is bitmap/full, B is sparse/hyper
//------------------------------------------------------------------------------

    #if GB_DISABLE
    #elif ( !GB_A_IS_PATTERN )

m4_divert(if_semiring_has_avx)
        //----------------------------------------------------------------------
        // saxpy5 method with vectors of length 8 for double, 16 for single
        //----------------------------------------------------------------------

        // AVX512F: vector registers are 512 bits, or 64 bytes, which can hold
        // 16 floats or 8 doubles.

        // # of bits in the type of C, for AVX2 and AVX512F (saxpy5 method only)
        #define GB_C_NBITS \
            GB_cn_bits

        #define GB_V16_512 (16 * GB_C_NBITS <= 512)
        #define GB_V8_512  ( 8 * GB_C_NBITS <= 512)
        #define GB_V4_512  ( 4 * GB_C_NBITS <= 512)

        #define GB_V16 GB_V16_512
        #define GB_V8  GB_V8_512
        #define GB_V4  GB_V4_512

        #if GB_COMPILER_SUPPORTS_AVX512F && GB_V4_512

            GB_TARGET_AVX512F static inline void GB_AxB_saxpy5_unrolled_avx512f
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
                #define GB_SEMIRING_HAS_AVX_IMPLEMENTATION 1
                #include "GB_AxB_saxpy5_unrolled.c"
            }

        #endif

        //----------------------------------------------------------------------
        // saxpy5 method with vectors of length 4 for double, 8 for single
        //----------------------------------------------------------------------

        // AVX2: vector registers are 256 bits, or 32 bytes, which can hold
        // 8 floats or 4 doubles.

        #define GB_V16_256 (16 * GB_C_NBITS <= 256)
        #define GB_V8_256  ( 8 * GB_C_NBITS <= 256)
        #define GB_V4_256  ( 4 * GB_C_NBITS <= 256)

        #undef  GB_V16
        #undef  GB_V8
        #undef  GB_V4

        #define GB_V16 GB_V16_256
        #define GB_V8  GB_V8_256
        #define GB_V4  GB_V4_256

        #if GB_COMPILER_SUPPORTS_AVX2 && GB_V4_256

            GB_TARGET_AVX2 static inline void GB_AxB_saxpy5_unrolled_avx2
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
                #define GB_SEMIRING_HAS_AVX_IMPLEMENTATION 1
                #include "GB_AxB_saxpy5_unrolled.c"
            }

        #endif
m4_divert(if_saxpy5_enabled)

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

    GrB_Info GB (_Asaxpy5B)
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
m4_divert(0)

//------------------------------------------------------------------------------
// GB_Asaxpy3B: C=A*B, C<M>=A*B, C<!M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

GrB_Info GB (_Asaxpy3B)
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
        return (GB (_Asaxpy3B_noM) (C, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    else if (!Mask_comp)
    {
        // C<M> = A*B
        return (GB (_Asaxpy3B_M) (C,
            M, Mask_struct, M_in_place, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    else
    {
        // C<!M> = A*B
        return (GB (_Asaxpy3B_notM) (C,
            M, Mask_struct, M_in_place, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_Asaxpy3B_M: C<M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#if ( !GB_DISABLE )

    GrB_Info GB (_Asaxpy3B_M)
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

    GrB_Info GB (_Asaxpy3B_noM)
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

    GrB_Info GB (_Asaxpy3B_notM)
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

