//------------------------------------------------------------------------------
// GB_ew: ewise kernels for each built-in binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h"
#include "GB_ewise_kernels.h"
#include "GB_ew__include.h"

// operator:
#define GB_BINOP(z,x,y,i,j) z = x
#define GB_Z_TYPE uint64_t
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE uint64_t

// A matrix:
#define GB_A_TYPE uint64_t
#define GB_A2TYPE uint64_t
#define GB_DECLAREA(aij) uint64_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [(A_iso) ? 0 : (pA)]

// B matrix:
#define GB_B_TYPE uint64_t
#define GB_B2TYPE void
#define GB_DECLAREB(bij) uint64_t bij
#define GB_GETB(bij,Bx,pB,B_iso)

// C matrix:
#define GB_C_TYPE uint64_t

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_FIRST || GxB_NO_UINT64 || GxB_NO_FIRST_UINT64)

#include "GB_ewise_shared_definitions.h"

//------------------------------------------------------------------------------
// C = A+B, all 3 matrices dense
//------------------------------------------------------------------------------

GrB_Info GB (_Cewise_fulln__first_uint64)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #include "GB_ewise_fulln_template.c"
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// C = A*D, column scale with diagonal D matrix
//------------------------------------------------------------------------------

GrB_Info GB (_AxD__first_uint64)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix D,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_colscale_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C = D*B, row scale with diagonal D matrix
//------------------------------------------------------------------------------

GrB_Info GB (_DxB__first_uint64)
(
    GrB_Matrix C,
    const GrB_Matrix D,
    const GrB_Matrix B,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_rowscale_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseAdd: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB (_AaddB__first_uint64)
(
    GrB_Matrix C,
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_ek_slicing,
    const int A_nthreads,
    const int A_ntasks,
    const int64_t *restrict B_ek_slicing,
    const int B_nthreads,
    const int B_ntasks
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #define GB_IS_EWISEUNION 0
    #include "GB_add_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseUnion: C=A+B, C<M>=A+B, C<!M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB (_AunionB__first_uint64)
(
    GrB_Matrix C,
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_void *alpha_scalar_in,
    const GB_void *beta_scalar_in,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_ek_slicing,
    const int A_nthreads,
    const int A_ntasks,
    const int64_t *restrict B_ek_slicing,
    const int B_nthreads,
    const int B_ntasks
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_A_TYPE alpha_scalar = (*((GB_A_TYPE *) alpha_scalar_in)) ;
    GB_B_TYPE beta_scalar  = (*((GB_B_TYPE *) beta_scalar_in )) ;
    #define GB_IS_EWISEUNION 1
    #include "GB_add_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C=A.*B, C<M>=A.*B, or C<M!>=A.*B where C is sparse/hyper
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_08__first_uint64)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_emult_08_meta.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C<#> = A.*B when A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_02__first_uint64)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict Cp_kfirst,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_emult_02_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C<#> = A.*B when A is bitmap/full and B is sparse/hyper
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_03__first_uint64)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict Cp_kfirst,
    const int64_t *B_ek_slicing,
    const int B_ntasks,
    const int B_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_emult_03_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C<M> = A.*B, M sparse/hyper, A and B bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_04__first_uint64)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict Cp_kfirst,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_emult_04_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// eWiseMult: C=A.*B, C<M>=A.*B, C<!M>=A.*B where C is bitmap
//------------------------------------------------------------------------------

GrB_Info GB (_AemultB_bitmap__first_uint64)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads,
    const int C_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_emult_bitmap_template.c"
    return (GrB_SUCCESS) ;
    #endif
}
