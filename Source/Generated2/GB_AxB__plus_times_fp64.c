//------------------------------------------------------------------------------
// GB_AxB__plus_times_fp64.c: matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated1/ or Generated2/ folder, do not edit it
// (it is auto-generated from Generator/*).

#include "GB_dev.h"

#ifndef GBCOMPACT

#include "GB.h"
#include "GB_control.h"
#include "GB_bracket.h"
#include "GB_sort.h"
#include "GB_atomics.h"
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

// A'*B (dot2):        GB (_Adot2B__plus_times_fp64)
// A'*B (dot3):        GB (_Adot3B__plus_times_fp64)
// C+=A'*B (dot4):     GB (_Adot4B__plus_times_fp64)
// A*B (saxpy bitmap): GB (_AsaxbitB__plus_times_fp64)
// A*B (saxpy3):       GB (_Asaxpy3B__plus_times_fp64)
//     no mask:        GB (_Asaxpy3B_noM__plus_times_fp64)
//     mask M:         GB (_Asaxpy3B_M__plus_times_fp64)
//     mask !M:        GB (_Asaxpy3B_notM__plus_times_fp64)
// A*B (saxpy4):       GB (_Asaxpy4B__plus_times_fp64)
// A*B (saxpy5):       GB (_Asaxpy5B__plus_times_fp64)

// C type:     double
// A type:     double
// A pattern?  0
// B type:     double
// B pattern?  0

// Multiply: z = (x * y)
// Add:      cij += t
//    'any' monoid?  0
//    atomic?        1
//    OpenMP atomic? 1
//    identity:      0
//    terminal?      0
//    terminal condition: ;
// MultAdd:  z += (x * y)

#define GB_ATYPE \
    double

#define GB_BTYPE \
    double

#define GB_CTYPE \
    double

#define GB_ASIZE \
    sizeof (double)

#define GB_BSIZE \
    sizeof (double) 

#define GB_CSIZE \
    sizeof (double)

// # of bits in the type of C, for AVX2 and AVX512F
#define GB_CNBITS \
    64

// true for int64, uint64, float, double, float complex, and double complex 
#define GB_CTYPE_IGNORE_OVERFLOW \
    1

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA,A_iso) \
    double aik = GBX (Ax, pA, A_iso)

// true if values of A are not used
#define GB_A_IS_PATTERN \
    0 \

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB,B_iso) \
    double bkj = GBX (Bx, pB, B_iso)

// true if values of B are not used
#define GB_B_IS_PATTERN \
    0 \

// Gx [pG] = Ax [pA]
#define GB_LOADA(Gx,pG,Ax,pA,A_iso) \
    Gx [pG] = GBX (Ax, pA, A_iso)

// Gx [pG] = Bx [pB]
#define GB_LOADB(Gx,pG,Bx,pB,B_iso) \
    Gx [pG] = GBX (Bx, pB, B_iso)

#define GB_CX(p) \
    Cx [p]

// multiply operator
#define GB_MULT(z, x, y, i, k, j) \
    z = (x * y)

// cast from a real scalar (or 2, if C is complex) to the type of C
#define GB_CTYPE_CAST(x,y) \
    ((double) x)

// cast from a real scalar (or 2, if A is complex) to the type of A
#define GB_ATYPE_CAST(x,y) \
    ((double) x)

// multiply-add
#define GB_MULTADD(z, x, y, i, k, j) \
    z += (x * y)

// monoid identity value
#define GB_IDENTITY \
    0

// 1 if the identity value can be assigned via memset, with all bytes the same
#define GB_HAS_IDENTITY_BYTE \
    1

// identity byte, for memset
#define GB_IDENTITY_BYTE \
    0

// true if the monoid has a terminal value
#define GB_MONOID_IS_TERMINAL \
    0

// break if cij reaches the terminal value (dot product only)
#define GB_DOT_TERMINAL(cij) \
    ;

// simd pragma for dot-product loop vectorization
#define GB_PRAGMA_SIMD_DOT(cij) \
    GB_PRAGMA_SIMD_REDUCTION (+,cij)

// simd pragma for other loop vectorization
#define GB_PRAGMA_SIMD_VECTORIZE GB_PRAGMA_SIMD

// 1 for the PLUS_PAIR_(real) semirings, not for the complex case
#define GB_IS_PLUS_PAIR_REAL_SEMIRING \
    0

// 1 if the semiring is accelerated with AVX2 or AVX512f
#define GB_SEMIRING_HAS_AVX_IMPLEMENTATION \
    1

// declare the cij scalar (initialize cij to zero for PLUS_PAIR)
#define GB_CIJ_DECLARE(cij) \
    double cij

// Cx [pC] = cij
#define GB_PUTC(cij,p) \
    Cx [p] = cij

// Cx [p] = t
#define GB_CIJ_WRITE(p,t) \
    Cx [p] = t

// C(i,j) += t
#define GB_CIJ_UPDATE(p,t) \
    Cx [p] += t

// x + y
#define GB_ADD_FUNCTION(x,y) \
    x + y

// bit pattern for bool, 8-bit, 16-bit, and 32-bit integers
#define GB_CTYPE_BITS \
    0

// 1 if monoid update can skipped entirely (the ANY monoid)
#define GB_IS_ANY_MONOID \
    0

// 1 if monoid update is EQ
#define GB_IS_EQ_MONOID \
    0

// 1 if monoid update can be done atomically, 0 otherwise
#define GB_HAS_ATOMIC \
    1

// 1 if monoid update can be done with an OpenMP atomic update, 0 otherwise
#if GB_COMPILER_MSC
    /* MS Visual Studio only has OpenMP 2.0, with fewer atomics */
    #define GB_HAS_OMP_ATOMIC \
        1
#else
    #define GB_HAS_OMP_ATOMIC \
        1
#endif

// 1 for the ANY_PAIR_ISO semiring
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
    0

// 1 for the SECONDJ or SECONDJ1 multiply operator
#define GB_IS_SECONDJ_MULTIPLIER \
    0

// 1 for the FIRSTI1, FIRSTJ1, SECONDI1, or SECONDJ1 multiply operators
#define GB_OFFSET \
    0

// atomic compare-exchange
#define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
    GB_ATOMIC_COMPARE_EXCHANGE_64 (target, expected, desired)

// Hx [i] = t
#define GB_HX_WRITE(i,t) \
    Hx [i] = t

// Cx [p] = Hx [i]
#define GB_CIJ_GATHER(p,i) \
    Cx [p] = Hx [i]

// Cx [p] += Hx [i]
#define GB_CIJ_GATHER_UPDATE(p,i) \
    Cx [p] += Hx [i]

// Hx [i] += t
#define GB_HX_UPDATE(i,t) \
    Hx [i] += t

// memcpy (&(Cx [p]), &(Hx [i]), len)
#define GB_CIJ_MEMCPY(p,i,len) \
    memcpy (Cx +(p), Hx +(i), (len) * sizeof(double));

// disable this semiring and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_PLUS || GxB_NO_TIMES || GxB_NO_FP64 || GxB_NO_PLUS_FP64 || GxB_NO_TIMES_FP64 || GxB_NO_PLUS_TIMES_FP64)

//------------------------------------------------------------------------------
// GB_Adot2B: C=A'*B, C<M>=A'*B, or C<!M>=A'*B: dot product method, C is bitmap
//------------------------------------------------------------------------------

// if A_not_transposed is true, then C=A*B is computed where A is bitmap or full

GrB_Info GB (_Adot2B__plus_times_fp64)
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

GrB_Info GB (_Adot3B__plus_times_fp64)
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
// GB_Adot4B:  C+=A'*B: dense dot product (not used for ANY_PAIR_ISO)
//------------------------------------------------------------------------------

#if 1

    GrB_Info GB (_Adot4B__plus_times_fp64)
    (
        GrB_Matrix C,
        const GrB_Matrix A, int64_t *restrict A_slice, int naslice,
        const GrB_Matrix B, int64_t *restrict B_slice, int nbslice,
        const int nthreads,
        GB_Context Context
    )
    { 

double tdot4 = omp_get_wtime ( ) ;

        #if GB_DISABLE
        return (GrB_NO_VALUE) ;
        #else
//        #include "GB_AxB_dot4_meta.c"
{

//------------------------------------------------------------------------------
// GB_AxB_dot4_meta:  C+=A'*B via dot products, where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C+=A'*B where C is a dense matrix and computed in-place.  The monoid of the
// semiring matches the accum operator, and the type of C matches the ztype of
// accum.  That is, no typecasting can be done with C.

// The matrix C is the user input matrix.  C is not iso on output, but might
// iso on input, in which case the input iso scalar is cinput, and C->x has
// been expanded to non-iso.  If A and/or B are hypersparse, the iso value of C
// has been expanded, so that C->x is initialized.  Otherwise, C->x is not
// initialized.  Instead, each entry is initialized by the iso value in
// the GB_GET4C(cij,p) macro.  A and/or B can be iso.

#define GB_DOT4

// cij += A(k,i) * B(k,j)
#undef  GB_DOT
#define GB_DOT(k,pA,pB)                                                 \
{                                                                       \
    GB_DOT_TERMINAL (cij) ;         /* break if cij == terminal */      \
    GB_GETA (aki, Ax, pA, A_iso) ;          /* aki = A(k,i) */          \
    GB_GETB (bkj, Bx, pB, B_iso) ;          /* bkj = B(k,j) */          \
    GB_MULTADD (cij, aki, bkj, i, k, j) ;   /* cij += aki * bkj */      \
}

{ 

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    const int64_t cvlen = C->vlen ;

    const int64_t  *restrict Bp = B->p ;
    const int8_t   *restrict Bb = B->b ;
    const int64_t  *restrict Bh = B->h ;
    const int64_t  *restrict Bi = B->i ;
    const bool B_iso = B->iso ;
    const int64_t vlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;

    const int64_t  *restrict Ap = A->p ;
    const int8_t   *restrict Ab = A->b ;
    const int64_t  *restrict Ah = A->h ;
    const int64_t  *restrict Ai = A->i ;
    const bool A_iso = A->iso ;
    const int64_t avdim = A->vdim ;
    ASSERT (A->vlen == B->vlen) ;
    ASSERT (A->vdim == C->vlen) ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;

    #if GB_IS_ANY_MONOID
    #error "dot4 not supported for ANY monoids"
    #endif

    #if !GB_A_IS_PATTERN
    const GB_ATYPE *restrict Ax = (GB_ATYPE *) A->x ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_BTYPE *restrict Bx = (GB_BTYPE *) B->x ;
    #endif
          GB_CTYPE *restrict Cx = (GB_CTYPE *) C->x ;

    int ntasks = naslice * nbslice ;

    //--------------------------------------------------------------------------
    // if C is iso on input: get the iso scalar and convert C to non-iso
    //--------------------------------------------------------------------------

double t = omp_get_wtime ( ) ;

    const bool C_in_iso = C->iso ;
    const GB_CTYPE cinput = (C_in_iso) ? Cx [0] : GB_IDENTITY ;
    if (C_in_iso)
    { 
        // allocate but do not initialize C->x unless A or B are hypersparse
        GrB_Info info = GB_convert_any_to_non_iso (C, A_is_hyper || B_is_hyper,
            Context) ;
        if (info != GrB_SUCCESS)
        { 
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        ASSERT (!C->iso) ;
        Cx = (GB_CTYPE *) C->x ;
    }

t = omp_get_wtime ( ) - t ;
printf ("to non-iso time: %g\n", t) ;

    //--------------------------------------------------------------------------
    // C += A'*B
    //--------------------------------------------------------------------------

//     #include "GB_meta16_factory.c"
{

//------------------------------------------------------------------------------
// GB_meta16_factory: 16 cases of a method for A and B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All 16 cases are handled: A and B are sparse, hyper, bitmap, or full.

#define GB_META16

{
    if (A_is_sparse)
    {

        if (B_is_sparse)
        { 

            //------------------------------------------------------------------
            // both A and B are sparse
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else if (B_is_hyper)
        { 

            //------------------------------------------------------------------
            // A is sparse and B is hyper
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else if (B_is_bitmap)
        { 

            //------------------------------------------------------------------
            // A is sparse and B is bitmap
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // A is sparse and B is full
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1
            #include "GB_meta16_methods.c"

        }
    }
    else if (A_is_hyper)
    {
        if (B_is_sparse)
        { 

            //------------------------------------------------------------------
            // A is hyper and B is sparse
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else if (B_is_hyper)
        { 

            //------------------------------------------------------------------
            // both A and B are hyper
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else if (B_is_bitmap)
        { 

            //------------------------------------------------------------------
            // A is hyper and B is bitmap
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // A is hyper and B is full
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1
//            #include "GB_meta16_methods.c"
{

//------------------------------------------------------------------------------
// GB_meta16_methods: methods for GB_meta16_factory.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    // declare macros that depend on the sparsity of A and B
    #include "GB_meta16_definitions.h"

    // dot product methods
    #if defined ( GB_DOT4 )
//    #include "GB_AxB_dot4_template.c"
{

//------------------------------------------------------------------------------
// GB_AxB_dot4_template:  C+=A'*B via dot products, where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C+=A'*B where C is full and computed in-place.  The monoid of the semiring
// matches the accum operator, and the type of C matches the ztype of accum.

// The PAIR and FIRSTJ multiplicative operators are important special cases.

// The matrix C is the user input matrix.  C is not iso on output, but might
// iso on input, in which case the input iso scalar is cinput, and C->x has
// been expanded to non-iso, and initialized if A and/or B are hypersparse.
// A and/or B can be iso.

// MIN_FIRSTJ or MIN_FIRSTJ1 semirings:
#define GB_IS_MIN_FIRSTJ_SEMIRING (GB_IS_IMIN_MONOID && GB_IS_FIRSTJ_MULTIPLIER)
// MAX_FIRSTJ or MAX_FIRSTJ1 semirings:
#define GB_IS_MAX_FIRSTJ_SEMIRING (GB_IS_IMAX_MONOID && GB_IS_FIRSTJ_MULTIPLIER)
// GB_OFFSET is 1 for the MIN/MAX_FIRSTJ1 semirings, and 0 otherwise.

#if GB_IS_ANY_MONOID
#error "dot4 is not used for the ANY monoid"
#endif

#undef  GB_GET4C
#define GB_GET4C(cij,p) cij = (C_in_iso) ? cinput : Cx [p]

#if ((GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_BITMAP || GB_B_IS_FULL ))
{

    //--------------------------------------------------------------------------
    // C += A'*B where A and B are both bitmap/full
    //--------------------------------------------------------------------------

    // FUTURE: This method is not particularly efficient when both A and B are
    // bitmap/full.  A better method would use tiles to reduce memory traffic.

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int a_tid = tid / nbslice ;
        const int b_tid = tid % nbslice ;
        const int64_t kA_start = A_slice [a_tid] ;
        const int64_t kA_end   = A_slice [a_tid+1] ;
        const int64_t kB_start = B_slice [b_tid] ;
        const int64_t kB_end   = B_slice [b_tid+1] ;

        for (int64_t j = kB_start ; j < kB_end ; j++)
        {

            //------------------------------------------------------------------
            // get B(:,j) and C(:,j)
            //------------------------------------------------------------------

            const int64_t pC_start = j * cvlen ;
            const int64_t pB_start = j * vlen ;

            //------------------------------------------------------------------
            // C(:,j) += A'*B(:,j)
            //------------------------------------------------------------------

            for (int64_t i = kA_start ; i < kA_end ; i++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                const int64_t pA = i * vlen ;

                //--------------------------------------------------------------
                // get C(i,j)
                //--------------------------------------------------------------

                int64_t pC = i + pC_start ;     // C(i,j) is at Cx [pC]
                GB_CTYPE GB_GET4C (cij, pC) ;   // cij = Cx [pC]

                //--------------------------------------------------------------
                // C(i,j) += A (:,i)*B(:,j): a single dot product
                //--------------------------------------------------------------

                int64_t pB = pB_start ;

                #if ( GB_A_IS_FULL && GB_B_IS_FULL )
                {

                    //----------------------------------------------------------
                    // both A and B are full
                    //----------------------------------------------------------

                    #if GB_IS_PAIR_MULTIPLIER
                    { 
                        #if GB_IS_EQ_MONOID
                        // EQ_PAIR semiring
                        cij = (cij == 1) ;
                        #elif (GB_CTYPE_BITS > 0)
                        // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
                        // for bool, 8-bit, 16-bit, or 32-bit integer
                        uint64_t t = ((uint64_t) cij) + vlen ;
                        cij = (GB_CTYPE) (t & GB_CTYPE_BITS) ;
                        #elif GB_IS_PLUS_FC32_MONOID
                        // PLUS monoid for float complex
                        cij = GxB_CMPLXF (crealf (cij) + (float) vlen, 0) ;
                        #elif GB_IS_PLUS_FC64_MONOID
                        // PLUS monoid for double complex
                        cij = GxB_CMPLX (creal (cij) + (double) vlen, 0) ;
                        #else
                        // PLUS monoid for float, double, or 64-bit integers 
                        cij += (GB_CTYPE) vlen ;
                        #endif
                    }
                    #elif GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry
                        if (vlen > 0)
                        { 
                            int64_t k = GB_OFFSET ;
                            cij = GB_IMIN (cij, k) ;
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry
                        if (vlen > 0)
                        { 
                            int64_t k = vlen-1 + GB_OFFSET ;
                            cij = GB_IMAX (cij, k) ;
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_DOT (cij)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        { 
                            GB_DOT (k, pA+k, pB+k) ;    // cij += A(k,i)*B(k,j)
                        }
                    }
                    #endif

                }
                #elif ( GB_A_IS_FULL && GB_B_IS_BITMAP )
                {

                    //----------------------------------------------------------
                    // A is full and B is bitmap
                    //----------------------------------------------------------

                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry in B(:,j)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Bb [pB+k])
                            { 
                                cij = GB_IMIN (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry in B(:,j)
                        for (int64_t k = vlen-1 ; k >= 0 ; k--)
                        {
                            if (Bb [pB+k])
                            { 
                                cij = GB_IMAX (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_DOT (cij)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Bb [pB+k])
                            { 
                                GB_DOT (k, pA+k, pB+k) ; // cij += A(k,i)*B(k,j)
                            }
                        }
                    }
                    #endif

                }
                #elif ( GB_A_IS_BITMAP && GB_B_IS_FULL )
                {

                    //----------------------------------------------------------
                    // A is bitmap and B is full
                    //----------------------------------------------------------

                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry in A(:,i)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Ab [pA+k])
                            { 
                                cij = GB_IMIN (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry in A(:,i)
                        for (int64_t k = vlen-1 ; k >= 0 ; k--)
                        {
                            if (Ab [pA+k])
                            { 
                                cij = GB_IMAX (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_DOT (cij)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Ab [pA+k])
                            { 
                                GB_DOT (k, pA+k, pB+k) ; // cij += A(k,i)*B(k,j)
                            }
                        }
                    }
                    #endif

                }
                #elif ( GB_A_IS_BITMAP && GB_B_IS_BITMAP )
                {

                    //----------------------------------------------------------
                    // both A and B are bitmap
                    //----------------------------------------------------------

                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Ab [pA+k] && Bb [pB+k])
                            { 
                                cij = GB_IMIN (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry
                        for (int64_t k = vlen-1 ; k >= 0 ; k--)
                        {
                            if (Ab [pA+k] && Bb [pB+k])
                            { 
                                cij = GB_IMAX (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_DOT (cij)
                        for (int64_t k = 0 ; k < vlen ; k++)
                        {
                            if (Ab [pA+k] && Bb [pB+k])
                            { 
                                GB_DOT (k, pA+k, pB+k) ; // cij += A(k,i)*B(k,j)
                            }
                        }
                    }
                    #endif

                }
                #endif

                //--------------------------------------------------------------
                // save C(i,j)
                //--------------------------------------------------------------

                Cx [pC] = cij ;
            }
        }
    }
}
#elif ((GB_A_IS_SPARSE || GB_A_IS_HYPER) && (GB_B_IS_BITMAP || GB_B_IS_FULL ))
{

    //--------------------------------------------------------------------------
    // C += A'*B when A is sparse/hyper and B is bitmap/full
    //--------------------------------------------------------------------------

    // special cases: these methods are very fast, but cannot do not need
    // to be unrolled.
    #undef  GB_SPECIAL_CASE_OR_TERMINAL
    #define GB_SPECIAL_CASE_OR_TERMINAL \
       (   GB_IS_PAIR_MULTIPLIER        /* the multiply op is PAIR */       \
        || GB_IS_MIN_FIRSTJ_SEMIRING    /* min_firstj semiring */           \
        || GB_IS_MAX_FIRSTJ_SEMIRING    /* max_firstj semiring */           \
        || GB_MONOID_IS_TERMINAL        /* monoid has a terminal value */   \
        || GB_B_IS_PATTERN )            /* B is pattern-only */

    // Transpose B and unroll the innermost loop if this condition holds: A
    // must be sparse, B must be full, and no special semirings or operators
    // can be used.  The monoid must not be terminal.  These conditions are
    // known at compile time.
    #undef  GB_UNROLL
    #define GB_UNROLL \
        ( GB_A_IS_SPARSE && GB_B_IS_FULL && !( GB_SPECIAL_CASE_OR_TERMINAL ) )

    // If GB_UNROLL is true at compile-time, the simpler variant can still be
    // used, without unrolling, for any of these conditions:  (1) A is very
    // sparse (fewer entries than the size of the W workspace) or (2) B is iso.

    // The unrolled method does not allow B to be iso or pattern-only (such as
    // for the FIRST multiplicative operator.  If B is iso or pattern-only, the
    // dense matrix G = B' would be a single scalar, or its values would not be
    // accessed at all, so there is no benefit to computing G.

    #if GB_UNROLL
    const int64_t wp = (bvdim == 1) ? 0 : GB_IMIN (bvdim, 4) ;
    const int64_t anz = GB_nnz (A) ;
// printf ("C=A'*B anz %ld wp %ld\n", anz, wp) ;
    if (anz < wp * vlen || B_iso)
    #endif
    {

// printf ("C=A'*B no workspace\n") ;

        //----------------------------------------------------------------------
        // C += A'*B without workspace
        //----------------------------------------------------------------------

double t = omp_get_wtime ( ) ;

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < ntasks ; tid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            const int64_t kA_start = A_slice [tid] ;
            const int64_t kA_end   = A_slice [tid+1] ;

            //------------------------------------------------------------------
            // C+=A'*B where A is sparse/hyper and B is bitmap/full
            //------------------------------------------------------------------

            if (bvdim == 1)
            {

                //--------------------------------------------------------------
                // C += A'*B where C is a single vector
                //--------------------------------------------------------------

//              const int64_t pC_start = 0 ;
//              const int64_t pB = 0 ;
//              const int64_t j = 0 ;
                #define pC_start 0
                #define pB 0
                #define j 0

                for (int64_t kA = kA_start ; kA < kA_end ; kA++)
                {
                    // get A(:,i)
                    #if GB_A_IS_HYPER
                    const int64_t i = Ah [kA] ;
                    #else
                    const int64_t i = kA ;
                    #endif
                    int64_t pA = Ap [kA] ;
                    const int64_t pA_end = Ap [kA+1] ;
                    const int64_t ainz = pA_end - pA ;
                    // C(i) += A(:,i)'*B(:,0)
                    #include "GB_AxB_dot4_cij.c"
                }

                #undef pC_start
                #undef pB
                #undef j

            }
            else
            {

                //--------------------------------------------------------------
                // C += A'*B where C is a matrix
                //--------------------------------------------------------------

                for (int64_t kA = kA_start ; kA < kA_end ; kA++)
                {
                    // get A(:,i)
                    #if GB_A_IS_HYPER
                    const int64_t i = Ah [kA] ;
                    #else
                    const int64_t i = kA ;
                    #endif
                    int64_t pA = Ap [kA] ;
                    const int64_t pA_end = Ap [kA+1] ;
                    const int64_t ainz = pA_end - pA ;
                    // C(i,:) += A(:,i)'*B
                    for (int64_t j = 0 ; j < bvdim ; j++)
                    {
                        // get B(:,j) and C(:,j)
                        const int64_t pC_start = j * cvlen ;
                        const int64_t pB = j * vlen ;
                        // C(i,j) += A(:,i)'*B(:,j)
                        #include "GB_AxB_dot4_cij.c"
                    }
                }
            }
        }

t = omp_get_wtime ( ) - t ; printf ("tdot %g\n", t) ;

    }
    #if GB_UNROLL
    else
    {

        //----------------------------------------------------------------------
        // C += A'*B: with workspace W for transposing B, one panel at a time
        //----------------------------------------------------------------------

        size_t W_size = 0 ;
        GB_BTYPE *restrict W = NULL ;
        if (bvdim > 1)
        {
            W = GB_MALLOC_WORK (wp * vlen, GB_BTYPE, &W_size) ;
            if (W == NULL)
            { 
                // out of memory
                return (GrB_OUT_OF_MEMORY) ;
            }
        }

        for (int64_t j1 = 0 ; j1 < bvdim ; j1 += 4)
        {

            //------------------------------------------------------------------
            // C(:,j1:j2-1) += A * B (:,j1:j2-1) for a single panel
            //------------------------------------------------------------------

            const int64_t j2 = GB_IMIN (j1 + 4, bvdim) ;
            switch (j2 - j1)
            {

                default :
                case 1 :
                {

                    //----------------------------------------------------------
                    // C(:,j1:j2-1) is a single vector; use B(:,j1) in place
                    //----------------------------------------------------------

                    const GB_BTYPE *restrict G = Bx + j1 * vlen ;
                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic,1)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        // get the task descriptor
                        const int64_t kA_start = A_slice [tid] ;
                        const int64_t kA_end   = A_slice [tid+1] ;
                        for (int64_t i = kA_start ; i < kA_end ; i++)
                        {
                            // get A(:,i)
                            const int64_t pA = Ap [i] ;
                            const int64_t pA_end = Ap [i+1] ;
                            // cx [0] = C(i,j1)
                            GB_CTYPE cx [1] ;
                            GB_GET4C (cx [0], i + j1*cvlen) ;
                            // cx [0] += A (:,i)'*G
                            for (int64_t p = pA ; p < pA_end ; p++)
                            { 
                                // aki = A(k,i)
                                const int64_t k = Ai [p] ;
                                GB_GETA (aki, Ax, p, A_iso) ;
                                // cx [0] += A(k,i)*G(k,0)
                                GB_MULTADD (cx [0], aki, G [k], i, k, j1) ;
                            }
                            // C(i,j1) = cx [0]
                            Cx [i + j1*cvlen] = cx [0] ;
                        }
                    }
                }
                break ;

                case 2 :
                {

                    //----------------------------------------------------------
                    // G = B(:,j1:j1+1) and convert to row-form
                    //----------------------------------------------------------

                    GB_BTYPE *restrict G = W ;
                    int64_t k ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (k = 0 ; k < vlen ; k++)
                    {
                        // G (k,0:1) = B (k,j1:j1+1)
                        const int64_t k2 = k << 1 ;
                        G [k2    ] = Bx [k + (j1    ) * vlen] ;
                        G [k2 + 1] = Bx [k + (j1 + 1) * vlen] ;
                    }

                    //----------------------------------------------------------
                    // C += A'*G where G is vlen-by-2 in row-form
                    //----------------------------------------------------------

                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic,1)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        // get the task descriptor
                        const int64_t kA_start = A_slice [tid] ;
                        const int64_t kA_end   = A_slice [tid+1] ;
                        for (int64_t i = kA_start ; i < kA_end ; i++)
                        {
                            // get A(:,i)
                            const int64_t pA = Ap [i] ;
                            const int64_t pA_end = Ap [i+1] ;
                            // cx [0:1] = C(i,j1:j1+1)
                            GB_CTYPE cx [2] ;
                            GB_GET4C (cx [0], i + (j1  )*cvlen) ;
                            GB_GET4C (cx [1], i + (j1+1)*cvlen) ;
                            // cx [0:1] += A (:,i)'*G
                            for (int64_t p = pA ; p < pA_end ; p++)
                            { 
                                // aki = A(k,i)
                                const int64_t k = Ai [p] ;
                                GB_GETA (aki, Ax, p, A_iso) ;
                                const int64_t k2 = k << 1 ;
                                // cx [0:1] += A(k,i)*G(k,0:1)
                                GB_MULTADD (cx [0], aki, G [k2],   i, k, j1) ;
                                GB_MULTADD (cx [1], aki, G [k2+1], i, k, j1+1) ;
                            }
                            // C(i,j1:j1+1) = cx [0:1]
                            Cx [i + (j1  )*cvlen] = cx [0] ;
                            Cx [i + (j1+1)*cvlen] = cx [1] ;
                        }
                    }
                }
                break ;

                case 3 :
                {

                    //----------------------------------------------------------
                    // G = B(:,j1:j1+2) and convert to row-form
                    //----------------------------------------------------------

                    GB_BTYPE *restrict G = W ;
                    int64_t k ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (k = 0 ; k < vlen ; k++)
                    {
                        // G (k,0:2) = B (k,j1:j1+2)
                        const int64_t k3 = k * 3 ;
                        G [k3    ] = Bx [k + (j1    ) * vlen] ;
                        G [k3 + 1] = Bx [k + (j1 + 1) * vlen] ;
                        G [k3 + 2] = Bx [k + (j1 + 2) * vlen] ;
                    }

                    //----------------------------------------------------------
                    // C += A'*G where G is vlen-by-3 in row-form
                    //----------------------------------------------------------

                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic,1)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        // get the task descriptor
                        const int64_t kA_start = A_slice [tid] ;
                        const int64_t kA_end   = A_slice [tid+1] ;
                        for (int64_t i = kA_start ; i < kA_end ; i++)
                        {
                            // get A(:,i)
                            const int64_t pA = Ap [i] ;
                            const int64_t pA_end = Ap [i+1] ;
                            // cx [0:2] = C(i,j1:j1+2)
                            GB_CTYPE cx [3] ;
                            GB_GET4C (cx [0], i + (j1  )*cvlen) ;
                            GB_GET4C (cx [1], i + (j1+1)*cvlen) ;
                            GB_GET4C (cx [2], i + (j1+2)*cvlen) ;
                            // cx [0:2] += A (:,i)'*G
                            for (int64_t p = pA ; p < pA_end ; p++)
                            { 
                                // aki = A(k,i)
                                const int64_t k = Ai [p] ;
                                GB_GETA (aki, Ax, p, A_iso) ;
                                const int64_t k3 = k * 3 ;
                                // cx [0:2] += A(k,i)*G(k,0:2)
                                GB_MULTADD (cx [0], aki, G [k3  ], i, k, j1) ;
                                GB_MULTADD (cx [1], aki, G [k3+1], i, k, j1+1) ;
                                GB_MULTADD (cx [2], aki, G [k3+2], i, k, j1+2) ;
                            }
                            // C(i,j1:j1+2) = cx [0:2]
                            Cx [i + (j1  )*cvlen] = cx [0] ;
                            Cx [i + (j1+1)*cvlen] = cx [1] ;
                            Cx [i + (j1+2)*cvlen] = cx [2] ;
                        }
                    }
                }
                break ;

                case 4 :
                {

                    //----------------------------------------------------------
                    // G = B(:,j1:j1+3) and convert to row-form
                    //----------------------------------------------------------

                    GB_BTYPE *restrict G = W ;
                    int64_t k ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (k = 0 ; k < vlen ; k++)
                    {
                        // G (k,0:3) = B (k,j1:j1+3)
                        const int64_t k4 = k << 2 ;
                        G [k4    ] = Bx [k + (j1    ) * vlen] ;
                        G [k4 + 1] = Bx [k + (j1 + 1) * vlen] ;
                        G [k4 + 2] = Bx [k + (j1 + 2) * vlen] ;
                        G [k4 + 3] = Bx [k + (j1 + 3) * vlen] ;
                    }

                    //----------------------------------------------------------
                    // C += A'*G where G is vlen-by-4 in row-form
                    //----------------------------------------------------------

                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic,1)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        // get the task descriptor
                        const int64_t kA_start = A_slice [tid] ;
                        const int64_t kA_end   = A_slice [tid+1] ;
                        for (int64_t i = kA_start ; i < kA_end ; i++)
                        {
                            // get A(:,i)
                            const int64_t pA = Ap [i] ;
                            const int64_t pA_end = Ap [i+1] ;
                            // cx [0:3] = C(i,j1:j1+3)
                            GB_CTYPE cx [4] ;
                            GB_GET4C (cx [0], i + (j1  )*cvlen) ;
                            GB_GET4C (cx [1], i + (j1+1)*cvlen) ;
                            GB_GET4C (cx [2], i + (j1+2)*cvlen) ;
                            GB_GET4C (cx [3], i + (j1+3)*cvlen) ;
                            // cx [0:3] += A (:,i)'*G
                            for (int64_t p = pA ; p < pA_end ; p++)
                            { 
                                // aki = A(k,i)
                                const int64_t k = Ai [p] ;
                                GB_GETA (aki, Ax, p, A_iso) ;
                                const int64_t k4 = k << 2 ;
                                // cx [0:3] += A(k,i)*G(k,0:3)
                                GB_MULTADD (cx [0], aki, G [k4  ], i, k, j1) ;
                                GB_MULTADD (cx [1], aki, G [k4+1], i, k, j1+1) ;
                                GB_MULTADD (cx [2], aki, G [k4+2], i, k, j1+2) ;
                                GB_MULTADD (cx [3], aki, G [k4+3], i, k, j1+3) ;
                            }
                            // C(i,j1:j1+3) = cx [0:3]
                            Cx [i + (j1  )*cvlen] = cx [0] ;
                            Cx [i + (j1+1)*cvlen] = cx [1] ;
                            Cx [i + (j1+2)*cvlen] = cx [2] ;
                            Cx [i + (j1+3)*cvlen] = cx [3] ;
                        }
                    }
                }
                break ;
            }
        }

        // free workspace
        GB_FREE_WORK (&W, W_size) ;
    }
    #endif

}
#elif ( (GB_A_IS_BITMAP || GB_A_IS_FULL) && (GB_B_IS_SPARSE || GB_B_IS_HYPER))
{

    //--------------------------------------------------------------------------
    // C += A'*B where A is bitmap/full and B is sparse/hyper
    //--------------------------------------------------------------------------

    // FUTURE: this can be unrolled, like the case above

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int64_t kB_start = B_slice [tid] ;
        const int64_t kB_end   = B_slice [tid+1] ;

        // TODO: reverse the order of this for loop and the for-i loop below
        for (int64_t kB = kB_start ; kB < kB_end ; kB++)
        {

            //------------------------------------------------------------------
            // get B(:,j) and C(:,j)
            //------------------------------------------------------------------

            #if GB_B_IS_HYPER
            const int64_t j = Bh [kB] ;
            #else
            const int64_t j = kB ;
            #endif
            const int64_t pC_start = j * cvlen ;
            const int64_t pB_start = Bp [kB] ;
            const int64_t pB_end = Bp [kB+1] ;
            const int64_t bjnz = pB_end - pB_start ;

            //------------------------------------------------------------------
            // C(:,j) += A'*B(:,j)
            //------------------------------------------------------------------

            for (int64_t i = 0 ; i < avdim ; i++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                const int64_t pA = i * vlen ;

                //--------------------------------------------------------------
                // get C(i,j)
                //--------------------------------------------------------------

                int64_t pC = i + pC_start ;     // C(i,j) is at Cx [pC]
                GB_CTYPE GB_GET4C (cij, pC) ;   // cij = Cx [pC]

                //--------------------------------------------------------------
                // C(i,j) += A (:,i)*B(:,j): a single dot product
                //--------------------------------------------------------------

                int64_t pB = pB_start ;

                #if ( GB_A_IS_FULL )
                {

                    //----------------------------------------------------------
                    // A is full and B is sparse/hyper
                    //----------------------------------------------------------

                    #if GB_IS_PAIR_MULTIPLIER
                    { 
                        #if GB_IS_EQ_MONOID
                        // EQ_PAIR semiring
                        cij = (cij == 1) ;
                        #elif (GB_CTYPE_BITS > 0)
                        // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
                        // for bool, 8-bit, 16-bit, or 32-bit integer
                        uint64_t t = ((uint64_t) cij) + bjnz ;
                        cij = (GB_CTYPE) (t & GB_CTYPE_BITS) ;
                        #elif GB_IS_PLUS_FC32_MONOID
                        // PLUS monoid for float complex
                        cij = GxB_CMPLXF (crealf (cij) + (float) bjnz, 0) ;
                        #elif GB_IS_PLUS_FC64_MONOID
                        // PLUS monoid for double complex
                        cij = GxB_CMPLX (creal (cij) + (double) bjnz, 0) ;
                        #else
                        // PLUS monoid for float, double, or 64-bit integers
                        cij += (GB_CTYPE) bjnz ;
                        #endif
                    }
                    #elif GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry in B(:,j)
                        if (bjnz > 0)
                        { 
                            int64_t k = Bi [pB] + GB_OFFSET ;
                            cij = GB_IMIN (cij, k) ;
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry in B(:,j)
                        if (bjnz > 0)
                        { 
                            int64_t k = Bi [pB_end-1] + GB_OFFSET ;
                            cij = GB_IMAX (cij, k) ;
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_DOT (cij)
                        for (int64_t p = pB ; p < pB_end ; p++)
                        { 
                            int64_t k = Bi [p] ;
                            GB_DOT (k, pA+k, p) ;   // cij += A(k,i)*B(k,j)
                        }
                    }
                    #endif

                }
                #else
                {

                    //----------------------------------------------------------
                    // A is bitmap and B is sparse/hyper
                    //----------------------------------------------------------

                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    {
                        // MIN_FIRSTJ semiring: take the first entry
                        for (int64_t p = pB ; p < pB_end ; p++)
                        {
                            int64_t k = Bi [p] ;
                            if (Ab [pA+k])
                            { 
                                cij = GB_IMIN (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #elif GB_IS_MAX_FIRSTJ_SEMIRING
                    {
                        // MAX_FIRSTJ semiring: take the last entry
                        for (int64_t p = pB_end-1 ; p >= pB ; p--)
                        {
                            int64_t k = Bi [p] ;
                            if (Ab [pA+k])
                            { 
                                cij = GB_IMAX (cij, k + GB_OFFSET) ;
                                break ;
                            }
                        }
                    }
                    #else
                    {
                        GB_PRAGMA_SIMD_DOT (cij)
                        for (int64_t p = pB ; p < pB_end ; p++)
                        {
                            int64_t k = Bi [p] ;
                            if (Ab [pA+k])
                            { 
                                GB_DOT (k, pA+k, p) ;   // cij += A(k,i)*B(k,j)
                            }
                        }
                    }
                    #endif

                }
                #endif

                //--------------------------------------------------------------
                // save C(i,j)
                //--------------------------------------------------------------

                Cx [pC] = cij ;
            }
        }
    }
}
#elif ( (GB_A_IS_SPARSE || GB_A_IS_HYPER) && (GB_B_IS_SPARSE || GB_B_IS_HYPER))
{

    //--------------------------------------------------------------------------
    // C+=A'*B where A and B are both sparse/hyper
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int a_tid = tid / nbslice ;
        const int b_tid = tid % nbslice ;
        const int64_t kA_start = A_slice [a_tid] ;
        const int64_t kA_end   = A_slice [a_tid+1] ;
        const int64_t kB_start = B_slice [b_tid] ;
        const int64_t kB_end   = B_slice [b_tid+1] ;

        //----------------------------------------------------------------------
        // C+=A'*B via dot products
        //----------------------------------------------------------------------

        for (int64_t kB = kB_start ; kB < kB_end ; kB++)
        {

            //------------------------------------------------------------------
            // get B(:,j) and C(:,j)
            //------------------------------------------------------------------

            #if GB_B_IS_HYPER
            const int64_t j = Bh [kB] ;
            #else
            const int64_t j = kB ;
            #endif
            const int64_t pC_start = j * cvlen ;
            const int64_t pB_start = Bp [kB] ;
            const int64_t pB_end = Bp [kB+1] ;
            const int64_t bjnz = pB_end - pB_start ;

            //------------------------------------------------------------------
            // C(:,j) += A'*B(:,j) where C is full
            //------------------------------------------------------------------

            for (int64_t kA = kA_start ; kA < kA_end ; kA++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                #if GB_A_IS_HYPER
                const int64_t i = Ah [kA] ;
                #else
                const int64_t i = kA ;
                #endif
                int64_t pA = Ap [kA] ;
                const int64_t pA_end = Ap [kA+1] ;
                const int64_t ainz = pA_end - pA ;

                //--------------------------------------------------------------
                // get C(i,j)
                //--------------------------------------------------------------

                int64_t pC = i + pC_start ;     // C(i,j) is at Cx [pC]
                GB_CTYPE GB_GET4C (cij, pC) ;   // cij = Cx [pC]

                //--------------------------------------------------------------
                // C(i,j) += A (:,i)*B(:,j): a single dot product
                //--------------------------------------------------------------

                int64_t pB = pB_start ;

                //----------------------------------------------------------
                // both A and B are sparse/hyper
                //----------------------------------------------------------

                // The MIN_FIRSTJ semirings are exploited, by terminating as
                // soon as any entry is found.  The MAX_FIRSTJ semirings are
                // not treated specially here.  They could be done with a
                // backwards traversal of the sparse vectors A(:,i) and
                // B(:,j).

                if (ainz == 0 || bjnz == 0 || 
                    Ai [pA_end-1] < Bi [pB_start] ||
                    Bi [pB_end-1] < Ai [pA])
                { 

                    //------------------------------------------------------
                    // A(:,i) and B(:,j) don't overlap, or are empty
                    //------------------------------------------------------

                }
                else if (ainz > 8 * bjnz)
                {

                    //------------------------------------------------------
                    // B(:,j) is very sparse compared to A(:,i)
                    //------------------------------------------------------

                    while (pA < pA_end && pB < pB_end)
                    {
                        int64_t ia = Ai [pA] ;
                        int64_t ib = Bi [pB] ;
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
                            // A(k,i) and B(k,j) are next entries to merge
                            GB_DOT (ia, pA, pB) ;   // cij += A(k,i)*B(k,j)
                            #if GB_IS_MIN_FIRSTJ_SEMIRING
                            break ;
                            #endif
                            pA++ ;
                            pB++ ;
                        }
                    }

                }
                else if (bjnz > 8 * ainz)
                {

                    //------------------------------------------------------
                    // A(:,i) is very sparse compared to B(:,j)
                    //------------------------------------------------------

                    while (pA < pA_end && pB < pB_end)
                    {
                        int64_t ia = Ai [pA] ;
                        int64_t ib = Bi [pB] ;
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
                            // A(k,i) and B(k,j) are next entries to merge
                            GB_DOT (ia, pA, pB) ;   // cij += A(k,i)*B(k,j)
                            #if GB_IS_MIN_FIRSTJ_SEMIRING
                            break ;
                            #endif
                            pA++ ;
                            pB++ ;
                        }
                    }

                }
                else
                {

                    //------------------------------------------------------
                    // A(:,i) and B(:,j) have about the same sparsity
                    //------------------------------------------------------

                    while (pA < pA_end && pB < pB_end)
                    {
                        int64_t ia = Ai [pA] ;
                        int64_t ib = Bi [pB] ;
                        if (ia < ib)
                        { 
                            // A(ia,i) appears before B(ib,j)
                            pA++ ;
                        }
                        else if (ib < ia)
                        { 
                            // B(ib,j) appears before A(ia,i)
                            pB++ ;
                        }
                        else // ia == ib == k
                        { 
                            // A(k,i) and B(k,j) are the entries to merge
                            GB_DOT (ia, pA, pB) ;   // cij += A(k,i)*B(k,j)
                            #if GB_IS_MIN_FIRSTJ_SEMIRING
                            break ;
                            #endif
                            pA++ ;
                            pB++ ;
                        }
                    }
                }

                //--------------------------------------------------------------
                // save C(i,j)
                //--------------------------------------------------------------

                Cx [pC] = cij ;
            }
        }
    }
}
#endif

#undef GB_IS_MIN_FIRSTJ_SEMIRING
#undef GB_IS_MAX_FIRSTJ_SEMIRING
#undef GB_GET4C
#undef GB_SPECIAL_CASE_OR_TERMINAL
#undef GB_UNROLL

}

    #elif defined ( GB_DOT3_PHASE1 )
    #include "GB_AxB_dot3_phase1_template.c"
    #elif defined ( GB_DOT3_PHASE2 )
    #include "GB_AxB_dot3_template.c"
    #elif defined ( GB_DOT2 )
    #include "GB_AxB_dot2_template.c"

    #else
    #error "method undefined"
    #endif

    // undefine the macros that define the A and B sparsity
    #undef GB_A_IS_SPARSE
    #undef GB_A_IS_HYPER
    #undef GB_A_IS_BITMAP
    #undef GB_A_IS_FULL
    #undef GB_B_IS_SPARSE
    #undef GB_B_IS_HYPER
    #undef GB_B_IS_BITMAP
    #undef GB_B_IS_FULL
}

}

        }
    }
    else if (A_is_bitmap)
    {
        if (B_is_sparse)
        { 

            //------------------------------------------------------------------
            // A is bitmap and B is sparse
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else if (B_is_hyper)
        { 

            //------------------------------------------------------------------
            // A is bitmap and B is hyper
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else if (B_is_bitmap)
        { 

            //------------------------------------------------------------------
            // both A and B are bitmap
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // A is bitmap and B is full
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1
            #include "GB_meta16_methods.c"

        }
    }
    else
    {
        if (B_is_sparse)
        { 

            //------------------------------------------------------------------
            // A is full and B is sparse
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else if (B_is_hyper)
        { 

            //------------------------------------------------------------------
            // A is full and B is hyper
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else if (B_is_bitmap)
        { 

            //------------------------------------------------------------------
            // A is full and B is bitmap
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0
            #include "GB_meta16_methods.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // both A and B are full
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1
            #include "GB_meta16_methods.c"

        }
    }
}

//------------------------------------------------------------------------------
// redefine macros for any sparity of A and B
//------------------------------------------------------------------------------

#undef GB_META16
#include "GB_meta16_definitions.h"

}

}

#undef GB_DOT
#undef GB_DOT4

}

tdot4 = omp_get_wtime ( ) - tdot4 ; printf ("tdot4 %g\n", tdot4) ;
        return (GrB_SUCCESS) ;
        #endif
    }

#endif

//------------------------------------------------------------------------------
// GB_AsaxbitB: C=A*B, C<M>=A*B, C<!M>=A*B: saxpy method, C is bitmap/full
//------------------------------------------------------------------------------

#include "GB_AxB_saxpy3_template.h"

GrB_Info GB (_AsaxbitB__plus_times_fp64)
(
    GrB_Matrix C,   // bitmap or full
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Context Context
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

#if 1

    GrB_Info GB (_Asaxpy4B__plus_times_fp64)
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
        GB_Context Context
    )
    { 
        #if GB_DISABLE
        return (GrB_NO_VALUE) ;
        #else
        #include "GB_AxB_saxpy4_template.c"
        return (GrB_SUCCESS) ;
        #endif
    }

#endif

//------------------------------------------------------------------------------
// GB_Asaxpy5B: C += A*B when C is full, A is bitmap/full, B is sparse/hyper
//------------------------------------------------------------------------------

#if 1

    #if GB_DISABLE
    #elif ( !GB_A_IS_PATTERN )

        //----------------------------------------------------------------------
        // saxpy5 method with vectors of length 8 for double, 16 for single
        //----------------------------------------------------------------------

        // AVX512F: vector registers are 512 bits, or 64 bytes, which can hold
        // 16 floats or 8 doubles.

        #define GB_V16_512 (16 * GB_CNBITS <= 512)
        #define GB_V8_512  ( 8 * GB_CNBITS <= 512)
        #define GB_V4_512  ( 4 * GB_CNBITS <= 512)

        #define GB_V16 GB_V16_512
        #define GB_V8  GB_V8_512
        #define GB_V4  GB_V4_512

        #if GB_SEMIRING_HAS_AVX_IMPLEMENTATION && GB_COMPILER_SUPPORTS_AVX512F \
            && GB_V4_512

            GB_TARGET_AVX512F static inline void GB_AxB_saxpy5_unrolled_avx512f
            (
                GrB_Matrix C,
                const GrB_Matrix A,
                const GrB_Matrix B,
                const int ntasks,
                const int nthreads,
                const int64_t *B_slice,
                GB_Context Context
            )
            {
                #include "GB_AxB_saxpy5_unrolled.c"
            }

        #endif

        //----------------------------------------------------------------------
        // saxpy5 method with vectors of length 4 for double, 8 for single
        //----------------------------------------------------------------------

        // AVX2: vector registers are 256 bits, or 32 bytes, which can hold
        // 8 floats or 4 doubles.

        #define GB_V16_256 (16 * GB_CNBITS <= 256)
        #define GB_V8_256  ( 8 * GB_CNBITS <= 256)
        #define GB_V4_256  ( 4 * GB_CNBITS <= 256)

        #undef  GB_V16
        #undef  GB_V8
        #undef  GB_V4

        #define GB_V16 GB_V16_256
        #define GB_V8  GB_V8_256
        #define GB_V4  GB_V4_256

        #if GB_SEMIRING_HAS_AVX_IMPLEMENTATION && GB_COMPILER_SUPPORTS_AVX2 \
            && GB_V4_256

            GB_TARGET_AVX2 static inline void GB_AxB_saxpy5_unrolled_avx2
            (
                GrB_Matrix C,
                const GrB_Matrix A,
                const GrB_Matrix B,
                const int ntasks,
                const int nthreads,
                const int64_t *B_slice,
                GB_Context Context
            )
            {
                #include "GB_AxB_saxpy5_unrolled.c"
            }

        #endif

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
            GB_Context Context
        )
        {
            #include "GB_AxB_saxpy5_unrolled.c"
        }

    #endif

    GrB_Info GB (_Asaxpy5B__plus_times_fp64)
    (
        GrB_Matrix C,
        const GrB_Matrix A,
        const GrB_Matrix B,
        const int ntasks,
        const int nthreads,
        const int64_t *B_slice,
        GB_Context Context
    )
    { 
        #if GB_DISABLE
        return (GrB_NO_VALUE) ;
        #else
        #include "GB_AxB_saxpy5_meta.c"
        return (GrB_SUCCESS) ;
        #endif
    }

#endif

//------------------------------------------------------------------------------
// GB_Asaxpy3B: C=A*B, C<M>=A*B, C<!M>=A*B: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

GrB_Info GB (_Asaxpy3B__plus_times_fp64)
(
    GrB_Matrix C,   // C<any M>=A*B, C sparse or hypersparse
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks, const int nfine, const int nthreads, const int do_sort,
    GB_Context Context
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
    if (M == NULL)
    {
        // C = A*B, no mask
        return (GB (_Asaxpy3B_noM__plus_times_fp64) (C, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Context)) ;
    }
    else if (!Mask_comp)
    {
        // C<M> = A*B
        return (GB (_Asaxpy3B_M__plus_times_fp64) (C,
            M, Mask_struct, M_in_place, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Context)) ;
    }
    else
    {
        // C<!M> = A*B
        return (GB (_Asaxpy3B_notM__plus_times_fp64) (C,
            M, Mask_struct, M_in_place, A, B,
            SaxpyTasks, ntasks, nfine, nthreads, do_sort, Context)) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_Asaxpy3B_M: C<M>=A*Bi: saxpy method (Gustavson + Hash)
//------------------------------------------------------------------------------

#if ( !GB_DISABLE )

    GrB_Info GB (_Asaxpy3B_M__plus_times_fp64)
    (
        GrB_Matrix C,   // C<M>=A*B, C sparse or hypersparse
        const GrB_Matrix M, const bool Mask_struct,
        const bool M_in_place,
        const GrB_Matrix A,
        const GrB_Matrix B,
        GB_saxpy3task_struct *restrict SaxpyTasks,
        const int ntasks, const int nfine, const int nthreads,
        const int do_sort,
        GB_Context Context
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

    GrB_Info GB (_Asaxpy3B_noM__plus_times_fp64)
    (
        GrB_Matrix C,   // C=A*B, C sparse or hypersparse
        const GrB_Matrix A,
        const GrB_Matrix B,
        GB_saxpy3task_struct *restrict SaxpyTasks,
        const int ntasks, const int nfine, const int nthreads,
        const int do_sort,
        GB_Context Context
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

    GrB_Info GB (_Asaxpy3B_notM__plus_times_fp64)
    (
        GrB_Matrix C,   // C<!M>=A*B, C sparse or hypersparse
        const GrB_Matrix M, const bool Mask_struct,
        const bool M_in_place,
        const GrB_Matrix A,
        const GrB_Matrix B,
        GB_saxpy3task_struct *restrict SaxpyTasks,
        const int ntasks, const int nfine, const int nthreads,
        const int do_sort,
        GB_Context Context
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

