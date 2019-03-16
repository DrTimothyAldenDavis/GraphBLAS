//------------------------------------------------------------------------------
// GB_AxB:  hard-coded C=A*B and C<M>=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Unless this file is Generator/GB_AxB.c, do not edit it (auto-generated)

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_AxB__semirings.h"

// The C=A*B semiring is defined by the following types and operators:

// A*B function (Gustavon):  GB_AgusB__plus_times_fp32
// A'*B function (dot):      GB_AdotB__plus_times_fp32
// A*B function (heap):      GB_AheapB__plus_times_fp32
// Z type:   float (the type of C)
// X type:   float (the type of x for z=mult(x,y))
// Y type:   float (the type of y for z=mult(x,y))
// handle flipxy: 0 (0 if mult(x,y) is commutative, 1 otherwise)
// Identity: 0 (where cij += identity does not change cij)
// Multiply: z = x * y
// Add:      cij += z
// Terminal: ;

#define GB_XTYPE \
    float
#define GB_YTYPE \
    float
#define GB_HANDLE_FLIPXY \
    0

#define GB_DOT_TERMINAL(cij) \
    ;

#define GB_MULTOP(z,x,y) \
    z = x * y

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA,asize)     \
    GB_atype aik = Ax [pA] ;     // SKIP if A pattern

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB,bsize)     \
    GB_btype bkj = Bx [pB] ;     // SKIP if B pattern

//------------------------------------------------------------------------------
// C<M>=A*B and C=A*B: gather/scatter saxpy-based method (Gustavson)
//------------------------------------------------------------------------------

#define GB_IDENTITY \
    0

// Sauna_Work [i] = identity
#define GB_CLEARW(Sauna_Work,i,identity,zsize)  \
    Sauna_Work [i] = identity ;

// Cx [p] = Sauna_Work [i]
#define GB_GATHERC(Cx,p,Sauna_Work,i,zsize) \
    Cx [p] = Sauna_Work [i] ;

// mult-add operation (no mask)
#define GB_MULTADD_NOMASK                   \
{                                           \
    /* Sauna_Work [i] += A(i,k) * B(k,j) */ \
    float t ;                            \
    GB_MULTIPLY (t, aik, bkj) ;             \
    Sauna_Work [i] += t ;             \
}

// mult-add operation (with mask)
#define GB_MULTADD_WITH_MASK                \
{                                           \
    if (mark == hiwater)                    \
    {                                       \
        /* first time C(i,j) seen */        \
        /* Sauna_Work [i] = A(i,k) * B(k,j) */      \
        GB_MULTIPLY (Sauna_Work [i], aik, bkj) ;    \
        Sauna_Mark [i] = hiwater + 1 ;      \
    }                                       \
    else                                    \
    {                                       \
        /* C(i,j) seen before, update it */ \
        /* Sauna_Work [i] += A(i,k) * B(k,j) */     \
        GB_MULTADD_NOMASK ;                 \
    }                                       \
}

GrB_Info GB_AgusB__plus_times_fp32
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const GrB_Matrix B,
    bool flipxy,
    GB_Sauna Sauna
)
{ 
    float *restrict Sauna_Work = Sauna->Sauna_Work ;
    float *restrict Cx = C->x ;
    GrB_Info info = GrB_SUCCESS ;
    #include "GB_AxB_Gustavson_flipxy.c"
    return (info) ;
}

//------------------------------------------------------------------------------
// C<M>=A'*B, C<!M>=A'*B or C=A'*B: dot product
//------------------------------------------------------------------------------

// t = aki*bkj
#define GB_DOT_MULT(aki,bkj)   \
    float t ;               \
    GB_MULTIPLY (t, aki, bkj) ;

// cij += t
#define GB_DOT_ADD             \
    cij += t ;

// cij = t
#define GB_DOT_COPY            \
    cij = t ;

// cij is not a pointer but a scalar; nothing to do
#define GB_DOT_REACQUIRE ;

// clear cij
#define GB_DOT_CLEAR           \
    cij = 0 ;

// save the value of C(i,j)
#define GB_DOT_SAVE            \
    Cx [cnz] = cij ;

GrB_Info GB_AdotB__plus_times_fp32
(
    GrB_Matrix *Chandle,
    const GrB_Matrix M,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    bool flipxy
)
{ 
    GrB_Matrix C = (*Chandle) ;
    float *restrict Cx = C->x ;
    float cij ;
    size_t bkj_size = B->type->size ;
    GrB_Info info = GrB_SUCCESS ;
    #include "GB_AxB_dot_flipxy.c"
    return (info) ;
}

//------------------------------------------------------------------------------
// C<M>=A*B and C=A*B: heap saxpy-based method
//------------------------------------------------------------------------------

#include "GB_heap.h"

// C(i,j) += A(i,k) * B(k,j)
#define GB_CIJ_MULTADD(cij, aik, bkj) \
{                                  \
    float t ;                   \
    GB_MULTIPLY (t, aik, bkj) ;    \
    cij += t ;               \
}

// cij is not a pointer but a scalar; nothing to do
#define GB_CIJ_REACQUIRE ;

// cij = identity
#define GB_CIJ_CLEAR \
    cij = 0 ;

// save the value of C(i,j)
#define GB_CIJ_SAVE \
    Cx [cnz] = cij ;

GrB_Info GB_AheapB__plus_times_fp32
(
    GrB_Matrix *Chandle,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const GrB_Matrix B,
    bool flipxy,
    int64_t *restrict List,
    GB_pointer_pair *restrict pA_pair,
    GB_Element *restrict Heap,
    const int64_t bjnz_max
)
{ 
    GrB_Matrix C = (*Chandle) ;
    float *restrict Cx = C->x ;
    float cij ;
    int64_t cvlen = C->vlen ;
    GB_CIJ_CLEAR ;
    GrB_Info info = GrB_SUCCESS ;
    #include "GB_AxB_heap_flipxy.c"
    return (info) ;
}

#endif

