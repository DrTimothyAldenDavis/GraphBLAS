//------------------------------------------------------------------------------
// GB_jit_kernel_trans_bind2nd.c: Cx = op (A',x)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_ewise_shared_definitions.h"

// cij = op (aij, y)
#undef  GB_APPLY_OP
#define GB_APPLY_OP(pC,pA)                      \
{                                               \
    GB_DECLAREA (aij) ;                         \
    GB_GETA (aij, Ax, pA, false) ;              \
    GB_BINOP (Cx [pC], aij, y, 0, 0) ;          \
}

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GB_void *y_input,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GB_void *y_input,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 
    GB_Y_TYPE y = (*((const GB_Y_TYPE *) y_input)) ;
    #include "GB_transpose_template.c"
    return (GrB_SUCCESS) ;
}

