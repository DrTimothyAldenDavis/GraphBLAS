//------------------------------------------------------------------------------
// GB_jit_kernel_select_bitmap:  select bitmap JIT kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_select_shared_definitions.h"

GrB_Info GB_jit_kernel
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
) ;

GrB_Info GB_jit_kernel
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int nthreads
)
{ 
    #if GB_DEPENDS_ON_Y
    GB_Y_TYPE y = *((GB_Y_TYPE *) ythunk) ;
    #endif
    #include "GB_select_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

