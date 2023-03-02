//------------------------------------------------------------------------------
// GB_jit_kernel_ewise_full_accum.c: C += A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_ewise_shared_definitions.h"

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #include "GB_ewise_full_accum_template.c"
    return (GrB_SUCCESS) ;
}

