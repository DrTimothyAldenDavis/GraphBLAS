//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "GB_assignop_kernels.h"
#include "GB_aop__include.h"

// operator:
#define GB_BINOP(z,x,y,i,j) z = x
#define GB_Z_TYPE uint8_t
#define GB_X_TYPE uint8_t
#define GB_Y_TYPE uint8_t

// A matrix:
#define GB_A_TYPE uint8_t
#define GB_A2TYPE uint8_t
#define GB_DECLAREA(aij) uint8_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [(A_iso) ? 0 : (pA)]

// B matrix:
#define GB_B_TYPE uint8_t
#define GB_B2TYPE void
#define GB_DECLAREB(bij) uint8_t bij
#define GB_GETB(bij,Bx,pB,B_iso)

// C matrix:
#define GB_C_TYPE uint8_t

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_FIRST || GxB_NO_UINT8 || GxB_NO_FIRST_UINT8)

#include "GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C += B, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumB__first_uint8)
(
    GrB_Matrix C,
    const GrB_Matrix B,
    const int64_t *B_ek_slicing,
    const int B_ntasks,
    const int B_nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += b, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumb__first_uint8)
(
    GrB_Matrix C,
    const GB_void *p_bwork,
    const int nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    
    return (GrB_SUCCESS) ;
    #endif
}

