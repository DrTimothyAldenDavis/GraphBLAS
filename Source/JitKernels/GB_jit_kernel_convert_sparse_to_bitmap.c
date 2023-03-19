//------------------------------------------------------------------------------
// GB_jit_kernel_convert_sparse_to_bitmap.c: convert sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_kernel_shared_definitions.h"

// cij = op (aij)
#define GB_COPY(Axnew,pnew,Ax,p)                \
{                                               \
    /* aij = Ax [p] */                          \
    GB_DECLAREA (aij) ;                         \
    GB_GETA (aij, Ax, p, false) ;               \
    /* Cx [p] = unop (aij) */                   \
    GB_UNARYOP (Axnew [pnew], aij, i, j, y) ;   \
}

GrB_Info GB_jit_kernel
(
    // output:
    GB_void *Ax_new,
    int8_t *Ab,
    // input:
    const GrB_Matrix A,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_jit_kernel
(
    // output:
    GB_void *Ax_new,
    int8_t *Ab,
    // input:
    const GrB_Matrix A,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #include "GB_convert_sparse_to_bitmap_template.c"
    return (GrB_SUCCESS) ;
}

