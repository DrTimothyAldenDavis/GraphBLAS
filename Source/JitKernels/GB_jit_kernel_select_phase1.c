//------------------------------------------------------------------------------
// GB_jit_kernel_select_phase1:  select phase 1 JIT kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_select_shared_definitions.h"

GrB_Info GB_jit_kernel
(
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_jit_kernel
(
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #if GB_DEPENDS_ON_Y
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #endif
    #include "GB_select_entry_phase1_template.c"
    return (GrB_SUCCESS) ;
}

