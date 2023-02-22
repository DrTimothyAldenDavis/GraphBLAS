//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_saxpy4.c: JIT kernel for C+=A*B saxpy4 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C+=A*B: C is full, saxpy4 method

#include "GB_AxB_shared_definitions.h"
#include "GB_AxB_saxpy3_template.h"

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int ntasks,
    const int nthreads,
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *A_slice,
    const int64_t *H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int ntasks,
    const int nthreads,
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *A_slice,
    const int64_t *H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
)
{ 
    #include "GB_AxB_saxpy4_meta.c"
    return (GrB_SUCCESS) ;
}

