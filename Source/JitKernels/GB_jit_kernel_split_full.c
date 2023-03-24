//------------------------------------------------------------------------------
// GB_jit_kernel_split_full: split full A into a full tile C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_unop_shared_definitions.h"

// cij = op (aij)
#define GB_COPY(pC,pA) GB_UNOP (Cx, pC, Ax, pA, false, i, j, y)

GrB_Info GB_jit_kernel
(
    // input/output
    GrB_Matrix C,
    // input:
    GrB_Matrix A,
    int64_t avstart,
    int64_t aistart,
    const int C_nthreads
) ;

GrB_Info GB_jit_kernel
(
    // input/output
    GrB_Matrix C,
    // input:
    GrB_Matrix A,
    int64_t avstart,
    int64_t aistart,
    const int C_nthreads
)
{ 
    #include "GB_split_full_template.c"
    return (GrB_SUCCESS) ;
}
