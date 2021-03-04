// SPDX-License-Identifier: Apache-2.0

GrB_Info GB_Cdense_05d
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GB_void *p_cwork,     // scalar of type C->type
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads
) ;

GrB_Info GB_Cdense_06d
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const bool Mask_struct,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads
) ;

GrB_Info GB_Cdense_25
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads
) ;

GrB_Info GB_convert_s2b
(
    GrB_Matrix A,
    GB_void *GB_RESTRICT Ax_new_void,
    int8_t  *GB_RESTRICT Ab,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads
) ;

