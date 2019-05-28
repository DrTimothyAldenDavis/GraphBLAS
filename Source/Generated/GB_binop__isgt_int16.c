
//------------------------------------------------------------------------------
// GB_binop:  hard-coded functions for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Unless this file is Generator/GB_binop.c, do not edit it (auto-generated)

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_binop__include.h"

// C=binop(A,B) is defined by the following types and operators:

// A+B function (eWiseAdd):    GB_AaddB__isgt_int16
// A.*B function (eWiseMult):  GB_AemultB__isgt_int16
// A*D function (colscale):    GB_AxD__isgt_int16
// D*A function (rowscale):    GB_DxB__isgt_int16

// C type:   int16_t
// A type:   int16_t
// B type:   int16_t
// BinaryOp: cij = (aij > bij)

#define GB_ATYPE \
    int16_t

#define GB_BTYPE \
    int16_t

#define GB_CTYPE \
    int16_t

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA)  \
    int16_t aij = Ax [pA]

// bij = Bx [pB]
#define GB_GETB(bij,Bx,pB)  \
    int16_t bij = Bx [pB]

// cij = Ax [pA]
#define GB_COPY_A_TO_C(cij,Ax,pA) \
    cij = Ax [pA] ;

// cij = Bx [pB]
#define GB_COPY_B_TO_C(cij,Bx,pB) \
    cij = Bx [pB] ;

#define GB_CX(p) Cx [p]

// binary operator
#define GB_BINOP(z, x, y)   \
    z = (x > y) ;

// do the numerical phases of GB_add and GB_emult
#define GB_PHASE_2_OF_2

//------------------------------------------------------------------------------
// C = A*D, column scale with diagonal D matrix
//------------------------------------------------------------------------------

void GB_AxD__isgt_int16
(
    GrB_Matrix C,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix D, bool D_is_pattern,
    int nthreads
)
{ 
    int16_t *restrict Cx = C->x ;
    #include "GB_AxB_colscale_meta.c"
}

//------------------------------------------------------------------------------
// C = D*B, row scale with diagonal D matrix
//------------------------------------------------------------------------------

void GB_DxB__isgt_int16
(
    GrB_Matrix C,
    const GrB_Matrix D, bool D_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int nthreads
)
{ 
    int16_t *restrict Cx = C->x ;
    #include "GB_AxB_rowscale_meta.c"
}

//------------------------------------------------------------------------------
// eWiseAdd: C = A+B or C<M> = A+B
//------------------------------------------------------------------------------

void GB_AaddB__isgt_int16
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int ntasks,
    const int nthreads
)
{ 
    #include "GB_add_template.c"
}

//------------------------------------------------------------------------------
// eWiseMult: C = A.*B or C<M> = A.*B
//------------------------------------------------------------------------------

void GB_AemultB__isgt_int16
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int ntasks,
    const int nthreads
)
{ 
    #include "GB_emult_template.c"
}

#endif

