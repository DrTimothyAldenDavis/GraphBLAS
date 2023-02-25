//------------------------------------------------------------------------------
// GB_binop:  hard-coded functions for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "GB_ewise_kernels.h"
#include "GB_binop__include.h"

// operator:
#define GB_BINOP(z,x,y,i,j) z = 1

// A matrix:
#define GB_A_TYPE uint32_t
#define GB_A2TYPE uint32_t
#define GB_DECLAREA(aij) uint32_t aij
#define GB_GETA(aij,Ax,pA,A_iso)
#define GB_A_IS_PATTERN 1

// B matrix:
#define GB_B_TYPE uint32_t
#define GB_B2TYPE uint32_t
#define GB_DECLAREB(bij) uint32_t bij
#define GB_GETB(bij,Bx,pB,B_iso)
#define GB_B_IS_PATTERN 1

// C matrix:
#define GB_C_TYPE uint32_t

// do the numerical phases of GB_add and GB_emult
#define GB_PHASE_2_OF_2

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_PAIR || GxB_NO_UINT32 || GxB_NO_PAIR_UINT32)

#include "GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C = A+B, all 3 matrices dense
//------------------------------------------------------------------------------

void GB (_Cdense_ewise3_noaccum__pair_uint32)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #include "GB_dense_ewise3_noaccum_template.c"
}

//------------------------------------------------------------------------------
// C += B, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumB__pair_uint32)
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
    
    { 
        #include "GB_dense_subassign_23_template.c"
    }
    
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += b, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumb__pair_uint32)
(
    GrB_Matrix C,
    const GB_void *p_bwork,
    const int nthreads
)
{
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    
    { 
        // get the scalar b for C += b, of type GB_B_TYPE
        GB_B_TYPE bwork = (*((GB_B_TYPE *) p_bwork)) ;
        #include "GB_dense_subassign_22_template.c"
        return (GrB_SUCCESS) ;
    }
    
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseAdd: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB (_AaddB__pair_uint32)
(
    GrB_Matrix C,
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool is_eWiseUnion,
    const GB_void *alpha_scalar_in,
    const GB_void *beta_scalar_in,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    GB_Werk Werk
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    GB_WERK_DECLARE (B_ek_slicing, int64_t) ;
    GB_A_TYPE alpha_scalar ;
    GB_B_TYPE beta_scalar ;
    if (is_eWiseUnion)
    {
        alpha_scalar = (*((GB_A_TYPE *) alpha_scalar_in)) ;
        beta_scalar  = (*((GB_B_TYPE *) beta_scalar_in )) ;
    }
    #include "GB_add_template.c"
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
    #endif
}

