//------------------------------------------------------------------------------
// GB_sel:  hard-coded functions for selection operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// FIXME:
#define GB_DEBUG

#include "GB_select.h"
#include "GB_ek_slice.h"
#include "GB_sel__include.h"

// kind
#define GB_ENTRY_SELECTOR

#define GB_A_TYPE \
    int32_t

// test value of Ax [p]
#define GB_TEST_VALUE_OF_ENTRY(keep,p)                  \
    bool keep = (Ax [p] > thunk)

// Cx [pC] = Ax [pA], no typecast
#define GB_SELECT_ENTRY(Cx,pC,Ax,pA)                    \
    Cx [pC] = Ax [pA]

#include "GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// GB_sel_phase1
//------------------------------------------------------------------------------

void GB (_sel_phase1__gt_thunk_int32)
(
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const GB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #include "GB_select_entry_phase1_template.c"
}

//------------------------------------------------------------------------------
// GB_sel_phase2
//------------------------------------------------------------------------------

void GB (_sel_phase2__gt_thunk_int32)
(
    int64_t *restrict Ci,
    GB_void *restrict Cx_out,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const GB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    GB_A_TYPE *restrict Cx = (GB_A_TYPE *) Cx_out ;
    #include "GB_select_phase2.c"
}

//------------------------------------------------------------------------------
// GB_sel_bitmap
//------------------------------------------------------------------------------

void GB (_sel_bitmap__gt_thunk_int32)
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const GB_IndexUnaryOp op,
    const int nthreads
)
{ 
    #include "GB_bitmap_select_template.c"
}

