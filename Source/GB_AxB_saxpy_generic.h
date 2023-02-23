//------------------------------------------------------------------------------
// GB_AxB_saxpy_generic.h: definitions for GB_AxB_saxpy_generic
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_AXB_SAXPY_GENERIC_H
#define GB_AXB_SAXPY_GENERIC_H

GrB_Info GB_AxB_saxpy_generic
(
    GrB_Matrix C,                   // any sparsity
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,          // ignored if C is bitmap
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    const int saxpy_method,         // saxpy3 or bitmap method
    // for saxpy3 only:
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL if C is bitmap
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

// The methods below have the identical signature since are all constructed
// from Template/GB_saxpy_generic_method.c.

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_generic_*: C is sparse or hypersparse
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy3_generic_firsti64
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_firstj64
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_secondj64
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_firsti32
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_firstj32
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_secondj32
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_first
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_second
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_flipped
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B (flipped)
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxpy3_generic_unflipped
(
    GrB_Matrix C,                   // C is sparse or hypersparse
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks,
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_AxB_saxbit_generic_*: C is bitmap (never full)
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxbit_generic_firsti64
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_firstj64
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_secondj64
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_firsti32
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_firstj32
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_secondj32
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_first
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_second
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_flipped
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B (flipped)
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

GrB_Info GB_AxB_saxbit_generic_unflipped
(
    GrB_Matrix C,                   // C is bitmap
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL
    int ntasks,
    int nfine,
    int nthreads,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
) ;

#endif

