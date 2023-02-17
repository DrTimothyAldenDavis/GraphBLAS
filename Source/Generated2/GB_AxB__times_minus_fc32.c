//------------------------------------------------------------------------------
// GB_AxB__times_minus_fc32.c: matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "GB_AxB_kernels.h"
#include "GB_AxB__include2.h"

// semiring operators:
#define GB_MULTADD(z,a,b,i,k,j) { GxB_FC32_t x_op_y = GB_FC32_minus (a,b) ; z = GB_FC32_mul (z, x_op_y) ; }
#define GB_MULT(z,a,b,i,k,j)    z = GB_FC32_minus (a,b)
#define GB_ADD(z,zin,t)         z = GB_FC32_mul (zin, t)
#define GB_UPDATE(z,t)          z = GB_FC32_mul (z, t)
// identity: GxB_CMPLXF(1,0)

// A matrix, typecast to A2 for multiplier input
#define GB_A_ISO A_iso
#define GB_A_IS_PATTERN 0
#define GB_A_TYPE GxB_FC32_t
#define GB_A2TYPE GxB_FC32_t
#define GB_DECLAREA(aik) GxB_FC32_t aik
#define GB_GETA(aik,Ax,pA,A_iso) aik = Ax [(A_iso) ? 0 : (pA)]

// B matrix, typecast to B2 for multiplier input
#define GB_B_ISO B_iso
#define GB_B_IS_PATTERN 0
#define GB_B_TYPE GxB_FC32_t
#define GB_B2TYPE GxB_FC32_t
#define GB_DECLAREB(bkj) GxB_FC32_t bkj
#define GB_GETB(bkj,Bx,pB,B_iso) bkj = Bx [(B_iso) ? 0 : (pB)]

// C matrix
#define GB_C_ISO 0
#define GB_C_TYPE GxB_FC32_t
#define GB_PUTC(cij,Cx,p) Cx [p] = cij

// special case semirings:

// monoid properties:
#define GB_Z_TYPE GxB_FC32_t
#define GB_DECLARE_IDENTITY(z) GxB_FC32_t z = GxB_CMPLXF(1,0)
#define GB_DECLARE_IDENTITY_CONST(z) const GxB_FC32_t z = GxB_CMPLXF(1,0)

#define GB_Z_ATOMIC_BITS 64
#define GB_Z_ATOMIC_TYPE uint64_t
#define GB_Z_HAS_ATOMIC_UPDATE 1

#define GB_Z_IGNORE_OVERFLOW 1

// special case multipliers:

// disable this semiring and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_TIMES || GxB_NO_MINUS || GxB_NO_FC32 || GxB_NO_TIMES_FC32 || GxB_NO_MINUS_FC32 || GxB_NO_TIMES_MINUS_FC32)

// finalize anything not yet defined
#include "GB_AxB_shared_definitions.h"

//------------------------------------------------------------------------------
// GB_Adot2B: C=A'*B, C<M>=A'*B, or C<!M>=A'*B: dot product method, C is bitmap
//------------------------------------------------------------------------------

// if A_not_transposed is true, then C=A*B is computed where A is bitmap or full

GrB_Info GB (_Adot2B__times_minus_fc32)
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

GrB_Info GB (_Adot3B__times_minus_fc32)
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

    GrB_Info GB (_Adot4B__times_minus_fc32)
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
// GB_AsaxbitB: C=A*B, C<M>=A*B, C<!M>=A*B: saxpy method, C is bitmap only
//------------------------------------------------------------------------------

#include "GB_AxB_saxpy3_template.h"

GrB_Info GB (_AsaxbitB__times_minus_fc32)
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

    GrB_Info GB (_Asaxpy5B__times_minus_fc32)
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

GrB_Info GB (_Asaxpy3B__times_minus_fc32)
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
        return (GB (_Asaxpy3B_noM__times_minus_fc32) (C, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    else if (!Mask_comp)
    {
        // C<M> = A*B
        return (GB (_Asaxpy3B_M__times_minus_fc32) (C,
            M, Mask_struct, M_in_place, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    else
    {
        // C<!M> = A*B
        return (GB (_Asaxpy3B_notM__times_minus_fc32) (C,
            M, Mask_struct, M_in_place, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk)) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_Asaxpy3B_M: C<M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#if ( !GB_DISABLE )

    GrB_Info GB (_Asaxpy3B_M__times_minus_fc32)
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
// GB_Asaxpy3B_noM: C=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#if ( !GB_DISABLE )

    GrB_Info GB (_Asaxpy3B_noM__times_minus_fc32)
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
// GB_Asaxpy3B_notM: C<!M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#if ( !GB_DISABLE )

    GrB_Info GB (_Asaxpy3B_notM__times_minus_fc32)
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

