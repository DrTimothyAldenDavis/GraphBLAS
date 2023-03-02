//------------------------------------------------------------------------------
// GB_jit_kernel_emult_02.c: C<#M>=A.*B, for emult_02
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_ewise_shared_definitions.h"

GrB_Info GB_jit_kernel
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
) ;

GrB_Info GB_jit_kernel
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
    #include "GB_emult_02_template.c"
    return (GrB_SUCCESS) ;
}

