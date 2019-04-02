//------------------------------------------------------------------------------
// GB_unaryop:  hard-coded functions for each built-in unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated)

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_unaryop__include.h"

// C=unop(A) is defined by the following types and operators:

// op(A)  function:  GB_unop__ainv_fp32_int8
// op(A') function:  GB_tran__ainv_fp32_int8

// C type:   float
// A type:   int8_t
// cast:     float cij = aij
// unaryop:  cij = -aij

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA)  \
    int8_t aij = Ax [pA]

#define GB_CX(p) Cx [p]

// unary operator
#define GB_OP(z, x)   \
    z = -x ;

// casting
#define GB_CASTING(z, x)   \
    float z = x ;

//------------------------------------------------------------------------------
// Cx = op(cast(Ax)), apply a unary operator
//------------------------------------------------------------------------------

void GB_unop__ainv_fp32_int8
(
    float *restrict Cx,
    int8_t *restrict Ax,
    int64_t anz,
    int nthreads
)
{ 
    #include "GB_unaryop_apply_op.c"
}

//------------------------------------------------------------------------------
// C = op(cast(A')), transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

void GB_tran__ainv_fp32_int8
(
    int64_t *restrict Cp,
    int64_t *restrict Ci,
    float *restrict Cx,
    const GrB_Matrix A
)
{ 
    int8_t *restrict Ax = A->x ;
    #include "GB_unaryop_transpose_op.c"
}

#endif

