//------------------------------------------------------------------------------
// GB_jit_kernel_apply_bind2nd.c: Cx = op (A,y)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_ewise_shared_definitions.h"

GrB_Info GB_jit_kernel
(
    GB_void *Cx_output,         // Cx and Ax may be aliased
    const GB_void *Ax_input,
    const GB_void *y_input,
    const int8_t *restrict Ab,
    int64_t anz,
    int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GB_void *Cx_output,         // Cx and Ax may be aliased
    const GB_void *Ax_input,
    const GB_void *y_input,
    const int8_t *restrict Ab,
    int64_t anz,
    int nthreads
)
{ 
    #include "GB_apply_bind2nd_template.c"
    return (GrB_SUCCESS) ;
}

