//------------------------------------------------------------------------------
// GB_jit_kernel_split_sparse: split sparse A into a sparse tile C
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
    int64_t akstart,
    int64_t aistart,
    int64_t *restrict Wp,
    const int64_t *restrict C_ek_slicing,
    const int C_ntasks,
    const int C_nthreads
) ;

GrB_Info GB_jit_kernel
(
    // input/output
    GrB_Matrix C,
    // input:
    GrB_Matrix A,
    int64_t akstart,
    int64_t aistart,
    int64_t *restrict Wp,
    const int64_t *restrict C_ek_slicing,
    const int C_ntasks,
    const int C_nthreads
)
{ 
    #include "GB_split_sparse_template.c"
    return (GrB_SUCCESS) ;
}

