//------------------------------------------------------------------------------
// GB_jit_kernel_AxB_saxpy3.c: saxpy3 matrix multiply for a single semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_AxB_shared_definitions.h"
#define Mask_comp   GB_MASK_COMP
#define Mask_struct GB_MASK_STRUCT
#include "GB_AxB_saxpy3_template.h"

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,   // C<any M>=A*B, C sparse or hypersparse
    const GrB_Matrix M,
    const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks,
    const int nfine,
    const int nthreads,
    const int do_sort,
    GB_Werk Werk
) ;

GrB_Info GB_jit_kernel
(
    GrB_Matrix C,   // C<any M>=A*B, C sparse or hypersparse
    const GrB_Matrix M,
    const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks,
    const int nfine,
    const int nthreads,
    const int do_sort,
    GB_Werk Werk
)
{ 
    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
    #include "GB_AxB_saxpy3_template.c"
    return (GrB_SUCCESS) ;
}

