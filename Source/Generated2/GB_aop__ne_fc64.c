//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "GB_assignop_kernels.h"
#include "GB_aop__include.h"

// operator:
#define GB_BINOP(z,x,y,i,j) z = GB_FC64_ne (x, y)
#define GB_Z_TYPE bool
#define GB_X_TYPE GxB_FC64_t
#define GB_Y_TYPE GxB_FC64_t

// A matrix:
#define GB_A_TYPE GxB_FC64_t
#define GB_A2TYPE GxB_FC64_t
#define GB_DECLAREA(aij) GxB_FC64_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [(A_iso) ? 0 : (pA)]

// B matrix:
#define GB_B_TYPE GxB_FC64_t
#define GB_B2TYPE GxB_FC64_t
#define GB_DECLAREB(bij) GxB_FC64_t bij
#define GB_GETB(bij,Bx,pB,B_iso) bij = Bx [(B_iso) ? 0 : (pB)]

// C matrix:
#define GB_C_TYPE bool
#define GB_COPY_A_TO_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = (creal (Ax [(A_iso) ? 0 : (pA)]) != 0) || (cimag (Ax [(A_iso) ? 0 : (pA)]) != 0)
#define GB_COPY_B_TO_C(Cx,pC,Bx,pB,B_iso) Cx [pC] = (creal (Bx [(B_iso) ? 0 : (pB)]) != 0) || (cimag (Bx [(B_iso) ? 0 : (pB)]) != 0)
#define GB_CTYPE_IS_ATYPE 0
#define GB_CTYPE_IS_BTYPE 0

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_NE || GxB_NO_FC64 || GxB_NO_NE_FC64)

#include "GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C += B, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumB__ne_fc64)
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

GrB_Info GB (_Cdense_accumb__ne_fc64)
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

