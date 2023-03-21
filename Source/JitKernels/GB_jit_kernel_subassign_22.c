//------------------------------------------------------------------------------
// GB_jit_kernel_subassign_22.c: C += A where C is dense, A is sparse or dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

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
    const GrB_Matrix M,         // NULL
    // A matrix or scalar:
    const GrB_Matrix A,         // NULL
    const void *scalar,
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
    const GrB_Matrix M,         // NULL
    // A matrix or scalar:
    const GrB_Matrix A,         // NULL
    const void *scalar,
    GB_Werk Werk
)
{
    GB_Y_TYPE ywork = (*((GB_Y_TYPE *) scalar)) ;
    #include "GB_subassign_22_template.c"
    return (GrB_SUCCESS) ;
}

