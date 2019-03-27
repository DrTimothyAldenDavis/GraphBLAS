//------------------------------------------------------------------------------
// GB_AxB:  hard-coded functions for semiring: C<M>=A*B or A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Unless this file is Generator/GB_AxB.c, do not edit it (auto-generated)

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_AxB__include.h"

// The C=A*B semiring is defined by the following types and operators:

// A*B function (Gustavon):  GB_AgusB__min_lor_uint64
// A'*B function (dot):      GB_AdotB__min_lor_uint64
// A*B function (heap):      GB_AheapB__min_lor_uint64

// C type:   uint64_t
// A type:   uint64_t
// B type:   uint64_t

// Multiply: z = ((aik != 0) || (bkj != 0))
// Add:      cij = GB_IMIN (cij, z)
// MultAdd:  cij = GB_IMIN (cij, ((aik != 0) || (bkj != 0)))
// Identity: UINT64_MAX
// Terminal: if (cij == 0) break ;

#define GB_ATYPE \
    uint64_t

#define GB_BTYPE \
    uint64_t

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA) \
    uint64_t aik = Ax [pA]

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB) \
    uint64_t bkj = Bx [pB]

#define GB_CX(p) Cx [p]

// multiply operator
#define GB_MULT(z, x, y)        \
    z = ((x != 0) || (y != 0)) ;

// multiply-add
#define GB_MULTADD(z, x, y)     \
    z = GB_IMIN (z, ((x != 0) || (y != 0))) ;

// copy scalar
#define GB_COPY(z,x) z = x ;

// monoid identity value (Gustavson's method only, with no mask)
#define GB_IDENTITY \
    UINT64_MAX

// break if cij reaches the terminal value (dot product only)
#define GB_DOT_TERMINAL(cij) \
    if (cij == 0) break ;

// cij is not a pointer but a scalar; nothing to do
#define GB_CIJ_REACQUIRE(cij,cnz) ;

// save the value of C(i,j)
#define GB_CIJ_SAVE(cij) Cx [cnz] = cij ;

#define GB_SAUNA_WORK(i) Sauna_Work [i]

//------------------------------------------------------------------------------
// C<M>=A*B and C=A*B: gather/scatter saxpy-based method (Gustavson)
//------------------------------------------------------------------------------

GrB_Info GB_AgusB__min_lor_uint64
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    GB_Sauna Sauna
)
{ 
    uint64_t *restrict Sauna_Work = Sauna->Sauna_Work ;
    uint64_t *restrict Cx = C->x ;
    GrB_Info info = GrB_SUCCESS ;
    #include "GB_AxB_Gustavson_meta.c"
    return (info) ;
}

//------------------------------------------------------------------------------
// C<M>=A'*B, C<!M>=A'*B or C=A'*B: dot product
//------------------------------------------------------------------------------

GrB_Info GB_AdotB__min_lor_uint64
(
    GrB_Matrix *Chandle,
    const GrB_Matrix M, const bool Mask_comp,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern
)
{ 
    GrB_Matrix C = (*Chandle) ;
    uint64_t *restrict Cx = C->x ;
    uint64_t cij ;
    GrB_Info info = GrB_SUCCESS ;
    #define GB_SINGLE_PHASE
    #include "GB_AxB_dot_meta.c"
    #undef GB_SINGLE_PHASE
    return (info) ;
}

//------------------------------------------------------------------------------
// C<M>=A'*B, C<!M>=A'*B or C=A'*B: dot product (phase 2)
//------------------------------------------------------------------------------

GrB_Info GB_Adot2B__min_lor_uint64
(
    GrB_Matrix *Chandle,
    const GrB_Matrix M, const bool Mask_comp,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    const int64_t *restrict C_count_start,
    const int64_t *restrict C_count_end
)
{ 
    GrB_Matrix C = (*Chandle) ;
    uint64_t *restrict Cx = C->x ;
    uint64_t cij ;
    #define GB_PHASE_2_OF_2
    #include "GB_AxB_dot_meta.c"
    #undef GB_PHASE_2_OF_2
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// C<M>=A*B and C=A*B: heap saxpy-based method
//------------------------------------------------------------------------------

#include "GB_heap.h"

GrB_Info GB_AheapB__min_lor_uint64
(
    GrB_Matrix *Chandle,
    const GrB_Matrix M,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *restrict List,
    GB_pointer_pair *restrict pA_pair,
    GB_Element *restrict Heap,
    const int64_t bjnz_max
)
{ 
    GrB_Matrix C = (*Chandle) ;
    uint64_t *restrict Cx = C->x ;
    uint64_t cij ;
    int64_t cvlen = C->vlen ;
    GrB_Info info = GrB_SUCCESS ;
    #include "GB_AxB_heap_meta.c"
    return (info) ;
}

#endif

