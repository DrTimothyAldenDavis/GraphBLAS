//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_saxbit.c: saxbit matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_AxB_saxpy3_template.h"

GB_JIT_KERNEL_AXB_SAXBIT_PROTO (GB_jit_kernel) ;
GB_JIT_KERNEL_AXB_SAXBIT_PROTO (GB_jit_kernel)
{
    #include "GB_AxB_saxbit_template.c"
    return (GrB_SUCCESS) ;
}

