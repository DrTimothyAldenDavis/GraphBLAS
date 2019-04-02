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

// op(A)  function:  GB_unop
// op(A') function:  GB_tran

// C type:   GB_ctype
// A type:   GB_atype
// cast:     GB_CAST(cij,aij)
// unaryop:  GB_UNARYOP(cij,aij)

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA)  \
    GB_geta

#define GB_CX(p) Cx [p]

// unary operator
#define GB_OP(z, x)   \
    GB_UNARYOP(z, x) ;

// casting
#define GB_CASTING(z, x)   \
    GB_CAST(z, x) ;

//------------------------------------------------------------------------------
// Cx = op(cast(Ax)), apply a unary operator
//------------------------------------------------------------------------------

void GB_unop
(
    GB_ctype *restrict Cx,
    GB_atype *restrict Ax,
    int64_t anz,
    int nthreads
)
{ 
    #include "GB_unaryop_apply_op.c"
}

//------------------------------------------------------------------------------
// C = op(cast(A')), transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

void GB_tran
(
    int64_t *restrict Cp,
    int64_t *restrict Ci,
    GB_ctype *restrict Cx,
    const GrB_Matrix A
)
{ 
    GB_atype *restrict Ax = A->x ;
    #include "GB_unaryop_transpose_op.c"
}

#endif

