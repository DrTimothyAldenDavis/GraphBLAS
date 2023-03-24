//------------------------------------------------------------------------------
// GB_jit_kernel_concat_full: concatenate a full A into a full matrix C
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
    int64_t cvstart,
    GrB_Matrix A,
    const int A_nthreads
) ;

GrB_Info GB_jit_kernel
(
    // input/output
    GrB_Matrix C,
    // input:
    int64_t cistart,
    int64_t cvstart,
    GrB_Matrix A,
    const int A_nthreads
)
{ 
    #include "GB_concat_full_template.c"
    return (GrB_SUCCESS) ;
}

