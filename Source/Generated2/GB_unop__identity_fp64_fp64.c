//------------------------------------------------------------------------------
// GB_unop:  hard-coded functions for each built-in unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "GB_unop__include.h"

// C type:   double
// A type:   double
// cast:     double cij = aij
// unaryop:  cij = aij

#define GB_A_TYPE \
    double

#define GB_C_TYPE \
    double

// declare aij as atype
#define GB_DECLAREA(aij) \
    double aij

// aij = Ax [pA]
#define GB_GETA(aij,Ax,pA,A_iso) \
    aij = Ax [pA]

#define GB_CX(p) Cx [p]

// unary operator
#define GB_OP(z, x) \
    z = x ;

// casting
#define GB_CAST(z, aij) \
    double z = aij ;

// cij = op (aij)
#define GB_CAST_OP(pC,pA)           \
{                                   \
    /* aij = Ax [pA] */             \
    double aij ;              \
    aij = Ax [pA] ;   \
    /* Cx [pC] = op (cast (aij)) */ \
    double z = aij ;               \
    Cx [pC] = z ;        \
}

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_IDENTITY || GxB_NO_FP64)

#include "GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// C = op (cast (A')): transpose, typecast, and apply a unary operator
//------------------------------------------------------------------------------

GrB_Info GB (_unop_tran__identity_fp64_fp64)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_unop_transpose.c"
    return (GrB_SUCCESS) ;
    #endif
}

