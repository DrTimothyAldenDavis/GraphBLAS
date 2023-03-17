//------------------------------------------------------------------------------
// GB_jit_kernel_subassign_25.c:
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 25: C(:,:)<M,s> = A ; C is empty, M structural, A bitmap/as-if-full

// M:           present
// Mask_comp:   false
// Mask_struct: true
// C_replace:   effectively false (not relevant since C is empty)
// accum:       NULL
// A:           matrix
// S:           none

// C and M are sparse or hypersparse.  A can have any sparsity structure, even
// bitmap, but it must either be bitmap, or as-if-full.  M may be jumbled.  If
// so, C is constructed as jumbled.  C is reconstructed with the same structure
// as M and can have any sparsity structure on input.  The only constraint on C
// is nnz(C) is zero on input.

// C is iso if A is iso

#include "GB_subassign_shared_definitions.h"

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
    const GrB_Matrix A,
    const void *scalar,         // NULL
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
    const GrB_Matrix A,
    const void *scalar,         // NULL
    GB_Werk Werk
)
{
    #include "GB_subassign_25_template.c"
    return (GrB_SUCCESS) ;
}

