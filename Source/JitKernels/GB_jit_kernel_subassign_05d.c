//------------------------------------------------------------------------------
// GB_jit_kernel_subassign_05d.c:  C<M> = scalar, when C is dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 05d: C(:,:)<M> = scalar ; no S, C is dense

// M:           present
// Mask_comp:   false
// Mask_struct: true or false
// C_replace:   false
// accum:       NULL
// A:           scalar, already cast to C->type
// S:           none
// I:           NULL
// J:           NULL

// C can have any sparsity structure, but it must be entirely dense with
// all entries present.

GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel) ;
GB_JIT_KERNEL_SUBASSIGN_PROTO (GB_jit_kernel)
{
    GB_C_TYPE cwork = (*((GB_C_TYPE *) scalar)) ;
    #include "GB_subassign_05d_template.c"
    return (GrB_SUCCESS) ;
}

