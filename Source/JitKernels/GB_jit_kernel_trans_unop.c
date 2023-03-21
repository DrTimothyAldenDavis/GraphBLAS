//------------------------------------------------------------------------------
// GB_jit_kernel_transpose_unop.c: C = op (A') for unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// TODO: C=op(A') is used only for unary ops; extend it to index unary ops

#include "GB_unop_shared_definitions.h"

// cij = op (aij)
#define GB_APPLY_OP(pC,pA) GB_UNOP (Cx, pC, Ax, pA, A_iso, i, j, y)

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    #include "GB_transpose_template.c"
    return (GrB_SUCCESS) ;
}

