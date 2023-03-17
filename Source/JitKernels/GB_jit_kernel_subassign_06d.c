//------------------------------------------------------------------------------
// GB_jit_kernel_subassign_06d.c:  C<M> = scalar, when C is dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Method 06d: C(:,:)<A> = A ; no S, C is dense, M and A are aliased

// M:           present, and aliased to A
// Mask_comp:   false
// Mask_struct: true or false
// C_replace:   false
// accum:       NULL
// A:           matrix, and aliased to M
// S:           none

// C must be bitmap or as-if-full.  No entries are deleted and thus no zombies
// are introduced into C.  C can be hypersparse, sparse, bitmap, or full, and
// its sparsity structure does not change.  If C is hypersparse, sparse, or
// full, then the pattern does not change (all entries are present, and this
// does not change), and these cases can all be treated the same (as if full).
// If C is bitmap, new entries can be inserted into the bitmap C->b.

// C and A can have any sparsity structure.

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
    const GrB_Matrix M,         // aliased to A
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
    const GrB_Matrix M,         // aliased to A
    // A matrix or scalar:
    const GrB_Matrix A,
    const void *scalar,         // NULL
    GB_Werk Werk
)
{
    ASSERT (M == A) ;
    #include "GB_subassign_06d_template.c"
    return (GrB_SUCCESS) ;
}

