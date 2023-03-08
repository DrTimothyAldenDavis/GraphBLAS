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
#define GB_BINOP(z,x,y,i,j) z = GxB_CMPLXF(1,0)
#define GB_Z_TYPE GxB_FC32_t
#define GB_X_TYPE GxB_FC32_t
#define GB_Y_TYPE GxB_FC32_t

// A matrix:
#define GB_A_TYPE GxB_FC32_t
#define GB_A2TYPE void
#define GB_DECLAREA(aij) GxB_FC32_t aij
#define GB_GETA(aij,Ax,pA,A_iso)

// B matrix:
#define GB_B_TYPE GxB_FC32_t
#define GB_B2TYPE void
#define GB_DECLAREB(bij) GxB_FC32_t bij
#define GB_GETB(bij,Bx,pB,B_iso)

// C matrix:
#define GB_C_TYPE GxB_FC32_t

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_PAIR || GxB_NO_FC32 || GxB_NO_PAIR_FC32)

#include "GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C += B, accumulate a sparse matrix into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumB__pair_fc32)
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
        #include "GB_subassign_23_template.c"
    }
    
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C += b, accumulate a scalar into a dense matrix
//------------------------------------------------------------------------------

GrB_Info GB (_Cdense_accumb__pair_fc32)
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
        #include "GB_subassign_22_template.c"
        return (GrB_SUCCESS) ;
    }
    
    return (GrB_SUCCESS) ;
    #endif
}

