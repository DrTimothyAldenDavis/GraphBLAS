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

GB_accumop
GB_ztype
GB_xtype
GB_ytype
GB_atype
GB_copy_aij_to_y
GB_ctype
GB_copy_aij_to_c
GB_accum_y

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    GB_disable

#include "GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// C += A, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_23)
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
    m4_divert(if_C_dense_update)
    { 
        #include "GB_subassign_23_template.c"
    }
    m4_divert(0)
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += y, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_22)
(
    GrB_Matrix C,
    const GB_void *ywork_handle,
    const int nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    m4_divert(if_C_dense_update)
    { 
        // get the scalar ywork for C += ywork, of type GB_Y_TYPE
        GB_Y_TYPE ywork = (*((GB_Y_TYPE *) ywork_handle)) ;
        #include "GB_subassign_22_template.c"
        return (GrB_SUCCESS) ;
    }
    m4_divert(0)
    return (GrB_SUCCESS) ;
    #endif
}

