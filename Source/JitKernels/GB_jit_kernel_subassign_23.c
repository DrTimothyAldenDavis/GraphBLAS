//------------------------------------------------------------------------------
// GB_jit_kernel_subassign_23.c: C += A where C is dense, A is sparse or dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 23: C += A, where C is dense

// M:           NULL
// Mask_comp:   false
// Mask_struct: ignored
// C_replace:   false
// accum:       present
// A:           matrix
// S:           none
// I:           NULL
// J:           NULL

// C and A can have any sparsity structure, but C must be as-if-full.

GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel) ;
GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel)
{
    #include "GB_subassign_23_template.c"
    return (GrB_SUCCESS) ;
}

