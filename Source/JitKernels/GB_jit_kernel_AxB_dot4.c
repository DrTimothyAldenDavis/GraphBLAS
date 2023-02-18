//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_dot4.c: JIT kernel for C+=A'*B dot4 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C+=A'*B: dot product, C is full, dot4 method

#include "GB_AxB_shared_definitions.h"

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A, int64_t *restrict A_slice, int naslice,
    const GrB_Matrix B, int64_t *restrict B_slice, int nbslice,
    int nthreads, 
    GB_Werk Werk
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A, int64_t *restrict A_slice, int naslice,
    const GrB_Matrix B, int64_t *restrict B_slice, int nbslice,
    int nthreads,
    GB_Werk Werk
)
{ 
    #include "GB_AxB_dot4_meta.c"
    return (GrB_SUCCESS) ;
}

