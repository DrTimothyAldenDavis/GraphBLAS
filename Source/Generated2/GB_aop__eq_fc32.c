//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> += A

#include "GB.h"
#include "GB_control.h"
#include "GB_assignop_kernels.h"
#include "GB_aop__include.h"

#define GB_ACCUM_OP(z,x,y) z = GB_FC32_eq (x, y)
#define GB_Z_TYPE bool
#define GB_X_TYPE GxB_FC32_t
#define GB_Y_TYPE GxB_FC32_t
#define GB_A_TYPE GxB_FC32_t
#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) GxB_FC32_t ywork = Ax [(A_iso) ? 0 : (pA)]
#define GB_C_TYPE bool
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = (crealf (Ax [(A_iso) ? 0 : (pA)]) != 0) || (cimagf (Ax [(A_iso) ? 0 : (pA)]) != 0)
#define GB_ACCUMULATE_scalar(Cx,pC,ywork) GB_ACCUM_OP (Cx [pC], Cx [pC], ywork)

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_EQ || GxB_NO_FC32 || GxB_NO_EQ_FC32)

#include "GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// C += A, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_23__eq_fc32)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += y, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_22__eq_fc32)
(
    GrB_Matrix C,
    const GB_void *ywork_handle,
    const int nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    
    return (GrB_SUCCESS) ;
    #endif
}
