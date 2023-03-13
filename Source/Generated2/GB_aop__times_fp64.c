//------------------------------------------------------------------------------
// GB_aop:  assign/subassign kernels with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> += A

#include "GB.h"
#include "GB_control.h"
#include "GB_ek_slice.h"
#include "GB_aop__include.h"

// accum operator
#define GB_ACCUM_OP(z,x,y) z = (x) * (y)
#define GB_Z_TYPE double
#define GB_X_TYPE double
#define GB_Y_TYPE double
#define GB_ACCUMULATE_scalar(Cx,pC,ywork) GB_ACCUM_OP (Cx [pC], Cx [pC], ywork)
#define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso) double ywork = Ax [(A_iso) ? 0 : (pA)]

// A and C matrices
#define GB_A_TYPE double
#define GB_C_TYPE double
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = Ax [(A_iso) ? 0 : (pA)]
#define GB_COPY_scalar_to_C(pC,cwork) Cx [pC] = cwork
#define GB_AX_MASK(Ax,pA,asize) (Ax [pA] != 0)

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_TIMES || GxB_NO_FP64 || GxB_NO_TIMES_FP64)

#include "GB_subassign_shared_definitions.h"

#undef  GB_FREE_ALL 
#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// C += A, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_23__times_fp64)
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

GrB_Info GB (_subassign_22__times_fp64)
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

