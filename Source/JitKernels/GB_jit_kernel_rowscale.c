//------------------------------------------------------------------------------
// GB_jit_kernel_rowscale.c: C=D*B matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_ewise_shared_definitions.h"

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix D,
    const GrB_Matrix B,
    const int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix D,
    const GrB_Matrix B,
    const int nthreads
)
{ 
    #include "GB_rowscale_template.c"
    return (GrB_SUCCESS) ;
}

