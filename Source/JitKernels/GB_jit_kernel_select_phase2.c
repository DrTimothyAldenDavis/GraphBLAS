//------------------------------------------------------------------------------
// GB_jit_kernel_select_phase2:  select phase 2 JIT kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_kernel_shared_definitions.h"

GrB_Info GB_jit_kernel
(
    int64_t *restrict Ci,
    GB_void *restrict Cx_out,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_jit_kernel
(
    int64_t *restrict Ci,
    GB_void *restrict Cx_out,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    GB_A_TYPE *restrict Cx = (GB_A_TYPE *) Cx_out ;
    #if GB_DEPENDS_ON_Y
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #endif
    #include "GB_select_phase2.c"
    return (GrB_SUCCESS) ;
}

