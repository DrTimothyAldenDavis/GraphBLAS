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

#define GB_ACCUM_OP(z,x,y) z = GB_pow_uint8 (x, y)
#define GB_Z_TYPE uint8_t
#define GB_X_TYPE uint8_t
#define GB_Y_TYPE uint8_t
#define GB_A_TYPE uint8_t
#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) uint8_t ywork = Ax [(A_iso) ? 0 : (pA)]
#define GB_C_TYPE uint8_t
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = Ax [(A_iso) ? 0 : (pA)]
#define GB_ACCUMULATE_scalar(Cx,pC,ywork) GB_ACCUM_OP (Cx [pC], Cx [pC], ywork)

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_POW || GxB_NO_UINT8 || GxB_NO_POW_UINT8)

#include "GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// C += A, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_23__pow_uint8)
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
    
    { 
        #include "GB_subassign_23_template.c"
    }
    
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += y, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_22__pow_uint8)
(
    GrB_Matrix C,
    const GB_void *ywork_handle,
    const int nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    
    { 
        // get the scalar ywork for C += ywork, of type GB_Y_TYPE
        GB_Y_TYPE ywork = (*((GB_Y_TYPE *) ywork_handle)) ;
        #include "GB_subassign_22_template.c"
        return (GrB_SUCCESS) ;
    }
    
    return (GrB_SUCCESS) ;
    #endif
}

