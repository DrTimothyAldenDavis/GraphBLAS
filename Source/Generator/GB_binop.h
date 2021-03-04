// SPDX-License-Identifier: Apache-2.0
if_is_binop_subset
void GB_Cdense_ewise3_accum
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
) ;
endif_is_binop_subset

GrB_Info GB_Cdense_ewise3_noaccum
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
) ;

GrB_Info GB_Cdense_accumB
(
    GrB_Matrix C,
    const GrB_Matrix B,
    const int64_t *B_ek_slicing, const int B_ntasks, const int B_nthreads
) ;

GrB_Info GB_Cdense_accumb
(
    GrB_Matrix C,
    const GB_void *p_bwork,
    const int nthreads
) ;

if_binop_is_semiring_multiplier

GrB_Info GB_AxD
(
    GrB_Matrix C,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix D, bool D_is_pattern,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads
) ;

GrB_Info GB_DxB
(
    GrB_Matrix C,
    const GrB_Matrix D, bool D_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int nthreads
) ;

endif_binop_is_semiring_multiplier

GrB_Info GB_AaddB
(
    GrB_Matrix C,
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool Ch_is_Mh,
    const int64_t *GB_RESTRICT C_to_M,
    const int64_t *GB_RESTRICT C_to_A,
    const int64_t *GB_RESTRICT C_to_B,
    const GB_task_struct *GB_RESTRICT TaskList,
    const int C_ntasks,
    const int C_nthreads,
    GB_Context Context
) ;

GrB_Info GB_AemultB_01
(
    GrB_Matrix C,
    const int C_sparsity,
    const int ewise_method,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *GB_RESTRICT C_to_M,
    const int64_t *GB_RESTRICT C_to_A,
    const int64_t *GB_RESTRICT C_to_B,
    const GB_task_struct *GB_RESTRICT TaskList,
    const int C_ntasks,
    const int C_nthreads,
    GB_Context Context
) ;

GrB_Info GB_AemultB_02
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool flipxy,
    const int64_t *GB_RESTRICT Cp_kfirst,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads
) ;

GrB_Info GB_AemultB_03
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *GB_RESTRICT Cp_kfirst,
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads
) ;

GrB_Info GB_AemultB_bitmap
(
    GrB_Matrix C,
    const int ewise_method,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads,
    const int C_nthreads,
    GB_Context Context
) ;

if_binop_bind1st_is_enabled
GrB_Info GB_bind1st
(
    GB_void *Cx_output,
    const GB_void *x_input,
    const GB_void *Bx_input,
    const int8_t *GB_RESTRICT Ab,
    int64_t anz,
    int nthreads
) ;
endif_binop_bind1st_is_enabled

if_binop_bind2nd_is_enabled
GrB_Info GB_bind2nd
(
    GB_void *Cx_output,
    const GB_void *Ax_input,
    const GB_void *y_input,
    const int8_t *GB_RESTRICT Ab,
    int64_t anz,
    int nthreads
) ;
endif_binop_bind2nd_is_enabled

if_binop_bind1st_is_enabled
GrB_Info GB_bind1st_tran
(
    GrB_Matrix C,
    const GB_void *x_input,
    const GrB_Matrix A,
    int64_t *GB_RESTRICT *Workspaces,
    const int64_t *GB_RESTRICT A_slice,
    int nworkspaces,
    int nthreads
) ;
endif_binop_bind1st_is_enabled

if_binop_bind2nd_is_enabled
GrB_Info GB_bind2nd_tran
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GB_void *y_input,
    int64_t *GB_RESTRICT *Workspaces,
    const int64_t *GB_RESTRICT A_slice,
    int nworkspaces,
    int nthreads
) ;
endif_binop_bind2nd_is_enabled

