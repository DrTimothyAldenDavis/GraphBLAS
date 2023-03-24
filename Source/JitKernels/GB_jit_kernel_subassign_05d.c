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
// A:           scalar
// S:           none

// C can have any sparsity structure, but it must be entirely dense with
// all entries present.

GrB_Info GB_jit_kernel
(
    // input/output:
    GrB_Matrix C,
    // input:
    // I:
    const GrB_Index *I,         // NULL
    const int64_t ni,           // 0
    const int64_t nI,           // 0
    const int64_t Icolon [3],   // NULL
    // J:
    const GrB_Index *J,         // NULL
    const int64_t nj,           // 0
    const int64_t nJ,           // 0
    const int64_t Jcolon [3],   // NULL
    // mask M:
    const GrB_Matrix M,
    // A matrix or scalar:
    const GrB_Matrix A,         // NULL
    const void *scalar,         // of type C->type
    GB_Werk Werk
) ;

GrB_Info GB_jit_kernel
(
    // input/output:
    GrB_Matrix C,
    // input:
    // I:
    const GrB_Index *I,         // NULL
    const int64_t ni,           // 0
    const int64_t nI,           // 0
    const int64_t Icolon [3],   // NULL
    // J:
    const GrB_Index *J,         // NULL
    const int64_t nj,           // 0
    const int64_t nJ,           // 0
    const int64_t Jcolon [3],   // NULL
    // mask M:
    const GrB_Matrix M,
    // A matrix or scalar:
    const GrB_Matrix A,         // NULL
    const void *scalar,         // of type C->type
    GB_Werk Werk
)
{
    GB_C_TYPE cwork = (*((GB_C_TYPE *) scalar)) ;
    #include "GB_subassign_05d_template.c"
    return (GrB_SUCCESS) ;
}

