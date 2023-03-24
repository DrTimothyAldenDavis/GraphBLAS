//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_dot2n.c: JIT kernel for C<#M>=A*B dot2n method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<#M>=A*B: dot product, C is bitmap/full, dot2n method

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A, int64_t *restrict A_slice,
    const GrB_Matrix B, int64_t *restrict B_slice,
    int nthreads, int naslice, int nbslice
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A, int64_t *restrict A_slice,
    const GrB_Matrix B, int64_t *restrict B_slice,
    int nthreads, int naslice, int nbslice
)
{ 
    #define GB_A_NOT_TRANSPOSED
    #include "GB_AxB_dot2_meta.c"
    return (GrB_SUCCESS) ;
}

