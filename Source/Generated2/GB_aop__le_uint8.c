//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> += B

#include "GB.h"
#include "GB_control.h"
#include "GB_assignop_kernels.h"
#include "GB_aop__include.h"

// accum:
#define GB_BINOP(z,x,y,i,j) z = ((x) <= (y))
#define GB_Z_TYPE bool
#define GB_X_TYPE uint8_t
#define GB_Y_TYPE uint8_t

// B matrix:
#define GB_B_TYPE uint8_t
#define GB_B2TYPE uint8_t
#define GB_DECLAREB(bij) uint8_t bij
#define GB_GETB(bij,Bx,pB,B_iso) bij = Bx [(B_iso) ? 0 : (pB)]

// C matrix:
#define GB_C_TYPE bool

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_LE || GxB_NO_UINT8 || GxB_NO_LE_UINT8)

#include "GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C += B, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumB__le_uint8)
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

GrB_Info GB (_Cdense_accumb__le_uint8)
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

