//------------------------------------------------------------------------------
// GB_jit_kernel_concat_sparse: concatenate A into a sparse matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij)
#define GB_COPY(pC,pA,A_iso) GB_UNOP (Cx, pC, Ax, pA, A_iso, i, j, y)

GrB_Info GB_jit_kernel
(
    // input/output
    GrB_Matrix C,
    // input:
    int64_t cistart,
    GrB_Matrix A,
    int64_t *restrict W,
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_jit_kernel
(
    // input/output
    GrB_Matrix C,
    // input:
    int64_t cistart,
    GrB_Matrix A,
    int64_t *restrict W,
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #include "GB_concat_sparse_template.c"
    return (GrB_SUCCESS) ;
}

