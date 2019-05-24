
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

// A+B function (eWiseAdd):    GB_AaddB__ge_int32
// A.*B function (eWiseMult):  GB_AemultB__ge_int32
// A*D function (colscale):    GB_AxD__ge_int32
// D*A function (rowscale):    GB_DxB__ge_int32

// C type:   bool
// A type:   int32_t
// B type:   int32_t
// BinaryOp: cij = (aij >= bij)

#define GB_ATYPE \
    int32_t

#define GB_BTYPE \
    int32_t

#define GB_CTYPE \
    bool

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA)  \
    int32_t aij = Ax [pA]

// bij = Bx [pB]
#define GB_GETB(bij,Bx,pB)  \
    int32_t bij = Bx [pB]

// cij = Ax [pA]
#define GB_COPY_A_TO_C(cij,Ax,pA) \
    cij = Ax [pA] ;

// cij = Bx [pB]
#define GB_COPY_B_TO_C(cij,Bx,pB) \
    cij = Bx [pB] ;

#define GB_CX(p) Cx [p]

// binary operator
#define GB_BINOP(z, x, y)   \
    z = (x >= y) ;

// do the numerical phases of GB_add and GB_emult
#define GB_PHASE_2_OF_2

//------------------------------------------------------------------------------
// C = A*D, column scale with diagonal D matrix
//------------------------------------------------------------------------------

void GB_AxD__ge_int32
(
    GrB_Matrix C,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix D, bool D_is_pattern,
    int nthreads
)
{ 
    bool *restrict Cx = C->x ;
    #include "GB_AxB_colscale_meta.c"
}

//------------------------------------------------------------------------------
// C = D*B, row scale with diagonal D matrix
//------------------------------------------------------------------------------

void GB_DxB__ge_int32
(
    GrB_Matrix C,
    const GrB_Matrix D, bool D_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int nthreads
)
{ 
    bool *restrict Cx = C->x ;
    #include "GB_AxB_rowscale_meta.c"
}

//------------------------------------------------------------------------------
// C = A+B, eWiseAdd, with any mask M
//------------------------------------------------------------------------------

void GB_AaddB__ge_int32
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_comp,
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
// C = A.*B, eWiseMult, with any mask M
//------------------------------------------------------------------------------

void GB_AemultB__ge_int32
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    int nthreads
)
{ 
    #include "GB_emult_template.c"
}

#endif

