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

// op(A)  function:  GB_unop__abs_int16_fp32
// op(A') function:  GB_tran__abs_int16_fp32

// C type:   int16_t
// A type:   float
// cast:     int16_t cij ; GB_CAST_SIGNED(cij,aij,16)
// unaryop:  cij = GB_IABS (aij)

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA)  \
    float aij = Ax [pA]

#define GB_CX(p) Cx [p]

// unary operator
#define GB_OP(z, x)   \
    z = GB_IABS (x) ;

// casting
#define GB_CASTING(z, x)   \
    int16_t z ; GB_CAST_SIGNED(z,x,16) ;

//------------------------------------------------------------------------------
// Cx = op(cast(Ax)), apply a unary operator
//------------------------------------------------------------------------------

void GB_unop__abs_int16_fp32
(
    int16_t *restrict Cx,
    float *restrict Ax,
    int64_t anz,
    int nthreads
)
{ 
    #include "GB_unaryop_apply_op.c"
}

//------------------------------------------------------------------------------
// C = op(cast(A')), transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

void GB_tran__abs_int16_fp32
(
    int64_t *restrict Cp,
    int64_t *restrict Ci,
    int16_t *restrict Cx,
    const GrB_Matrix A
)
{ 
    float *restrict Ax = A->x ;
    #include "GB_unaryop_transpose_op.c"
}

#endif

