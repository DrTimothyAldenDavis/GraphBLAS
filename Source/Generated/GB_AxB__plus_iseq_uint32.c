





//------------------------------------------------------------------------------
// GB_AxB:  hard-coded functions for semiring: C<M>=A*B or A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated).

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "GB_ek_slice.h"
#include "GB_Sauna.h"
#include "GB_jappend.h"
#include "GB_bracket.h"
#include "GB_iterator.h"
#include "GB_sort.h"
#include "GB_saxpy3.h"
#include "GB_AxB__include.h"

// The C=A*B semiring is defined by the following types and operators:

// A*B function (Gustavon):  GB_AgusB__plus_iseq_uint32
// A'*B function (dot2):     GB_Adot2B__plus_iseq_uint32
// A'*B function (dot3):     GB_Adot3B__plus_iseq_uint32
// A*B function (heap):      GB_AheapB__plus_iseq_uint32
// A*B function (saxpy3):    GB_Asaxpy3B__plus_iseq_uint32

// C type:   uint32_t
// A type:   uint32_t
// B type:   uint32_t

// Multiply: z = (aik == bkj)
// Add:      cij += z
//           atomic?        1
//           OpenMP atomic? 1
// MultAdd:  cij += (aik == bkj)
// Identity: 0
// Terminal: ;

#define GB_ATYPE \
    uint32_t

#define GB_BTYPE \
    uint32_t

#define GB_CTYPE \
    uint32_t

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA) \
    uint32_t aik = Ax [pA]

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB) \
    uint32_t bkj = Bx [pB]

#define GB_CX(p) Cx [p]

// multiply operator
#define GB_MULT(z, x, y) \
    z = (x == y)

// multiply-add
#define GB_MULTADD(z, x, y) \
    z += (x == y)

// copy scalar
#define GB_COPY_C(z,x) z = x

// monoid identity value (Gustavson's method only, with no mask)
#define GB_IDENTITY \
    0

// break if cij reaches the terminal value (dot product only)
#define GB_DOT_TERMINAL(cij) \
    ;

// simd pragma for dot product
#define GB_DOT_SIMD \
    GB_PRAGMA_SIMD

// cij is not a pointer but a scalar; nothing to do
#define GB_CIJ_REACQUIRE(cij,cnz)

// declare the cij scalar
#define GB_CIJ_DECLARE(cij) \
    uint32_t cij

// save the value of C(i,j)
#define GB_CIJ_SAVE(cij,p) Cx [p] = cij

#define GB_SAUNA_WORK(i) Sauna_Work [i]

// For saxpy3:

// Cx [p] = t
#define GB_CIJ_WRITE(p,t) Cx [p] = t

// C(i,j) += t
#define GB_CIJ_UPDATE(p,t) \
    Cx [p] += t

// Cx [p] = Hx [i]
#define GB_CIJ_GATHER(p,i) Cx [p] = Hx [i]

// Hx [i] += t
#define GB_HX_UPDATE(i,t) \
    Hx [i] += t

// x + y
#define GB_ADD_FUNCTION(x,y) \
    x + y

// type with size of GB_CTYPE, and can be used in compare-and-swap
#define GB_CTYPE_PUN \
    uint32_t

// Hx [i] = t
#define GB_HX_WRITE(i,t) Hx [i] = t

// 1 if monoid update can be done atomically, 0 otherwise
#define GB_HAS_ATOMIC \
    1

// 1 if monoid update can be done with a #pragma omp atomic update, 0 otherwise
#define GB_HAS_OMP_ATOMIC \
    1

// memcpy (&(Cx [p]), &(Hx [i]), len)
#define GB_CIJ_MEMCPY(p,i,len) \
    memcpy (Cx +(p), Hx +(i), (len) * sizeof(uint32_t))

// disable this semiring and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_PLUS || GxB_NO_ISEQ || GxB_NO_UINT32 || GxB_NO_PLUS_UINT32 || GxB_NO_ISEQ_UINT32 || GxB_NO_PLUS_ISEQ_UINT32)

//------------------------------------------------------------------------------
// C<M>=A*B and C=A*B: gather/scatter saxpy-based method (Gustavson)
//------------------------------------------------------------------------------

GrB_Info GB_AgusB__plus_iseq_uint32
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    GB_Sauna Sauna
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    uint32_t *GB_RESTRICT Sauna_Work = Sauna->Sauna_Work ;
    uint32_t *GB_RESTRICT Cx = C->x ;
    GrB_Info info = GrB_SUCCESS ;
    #include "GB_AxB_Gustavson_meta.c"
    return (info) ;
    #endif
}

//------------------------------------------------------------------------------
// C=A'*B or C<!M>=A'*B: dot product (phase 2)
//------------------------------------------------------------------------------

GrB_Info GB_Adot2B__plus_iseq_uint32
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix *Aslice, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *GB_RESTRICT B_slice,
    int64_t *GB_RESTRICT *C_counts,
    int nthreads, int naslice, int nbslice
)
{ 
    // C<M>=A'*B now uses dot3
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #define GB_PHASE_2_OF_2
    #include "GB_AxB_dot2_meta.c"
    #undef GB_PHASE_2_OF_2
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<M>=A'*B: masked dot product method (phase 2)
//------------------------------------------------------------------------------

GrB_Info GB_Adot3B__plus_iseq_uint32
(
    GrB_Matrix C,
    const GrB_Matrix M,
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
// C=A*B: saxpy3 method
//------------------------------------------------------------------------------

GrB_Info GB_Asaxpy3B__plus_iseq_uint32
(
    GrB_Matrix *Chandle,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    GB_saxpy3task_struct *GB_RESTRICT *TaskList_handle,
    void *Work [3], size_t Worksize [3],
    const int ntasks,
    const int nfine,
    const int nthreads,
    GB_Context Context
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    // get copies of these pointers for GB_FREE_ALL
    GB_saxpy3task_struct *GB_RESTRICT TaskList = (*TaskList_handle) ;
    GrB_Matrix C = (*Chandle) ;
    int64_t *Hi_all = Work [0] ;
    int64_t *Hf_all = Work [1] ;
    GB_void *Hx_all = Work [2] ;
    size_t Hi_size_total = Worksize [0] ;
    size_t Hf_size_total = Worksize [1] ;
    size_t Hx_size_total = Worksize [2] ;
    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    #include "GB_AxB_saxpy3_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<M>=A*B and C=A*B: heap saxpy-based method
//------------------------------------------------------------------------------

#include "GB_heap.h"

GrB_Info GB_AheapB__plus_iseq_uint32
(
    GrB_Matrix *Chandle,
    const GrB_Matrix M,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *GB_RESTRICT List,
    GB_pointer_pair *GB_RESTRICT pA_pair,
    GB_Element *GB_RESTRICT Heap,
    const int64_t bjnz_max
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GrB_Matrix C = (*Chandle) ;
    uint32_t *GB_RESTRICT Cx = C->x ;
    uint32_t cij ;
    int64_t cvlen = C->vlen ;
    GrB_Info info = GrB_SUCCESS ;
    #include "GB_AxB_heap_meta.c"
    return (info) ;
    #endif
}

#endif

