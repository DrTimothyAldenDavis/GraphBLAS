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
GB_binaryop
GB_ztype
GB_xtype
GB_ytype
GB_op_is_second

// A matrix:
GB_atype
GB_a2type
GB_declarea
GB_geta

// B matrix:
GB_btype
GB_b2type
GB_declareb
GB_getb

// C matrix:
GB_ctype
GB_copy_a_to_c
GB_copy_b_to_c
GB_ctype_is_atype
GB_ctype_is_btype

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    GB_disable

#include "GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C += B, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumB)
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
    m4_divert(if_C_dense_update)
    { 
        #include "GB_subassign_23_template.c"
    }
    m4_divert(0)
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += b, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumb)
(
    GrB_Matrix C,
    const GB_void *p_bwork,
    const int nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    m4_divert(if_C_dense_update)
    { 
        // get the scalar b for C += b, of type GB_B_TYPE
        GB_B_TYPE bwork = (*((GB_B_TYPE *) p_bwork)) ;
        #include "GB_subassign_22_template.c"
        return (GrB_SUCCESS) ;
    }
    m4_divert(0)
    return (GrB_SUCCESS) ;
    #endif
}

