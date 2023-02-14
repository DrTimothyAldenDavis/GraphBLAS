//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_dot3.c: JIT kernel for C<M>=A'*B dot3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M>=A'*B: masked dot product, C and M are both sparse or both hyper

#include "GB_AxB_shared_definitions.h"

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_task_struct *restrict TaskList,
    const int ntasks,
    const int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_task_struct *restrict TaskList,
    const int ntasks,
    const int nthreads
)
{ 
    #include "GB_AxB_dot3_meta.c"
    return (GrB_SUCCESS) ;
}

