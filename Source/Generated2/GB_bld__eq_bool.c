

//------------------------------------------------------------------------------
// GB_bld:  hard-coded functions for builder methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated2/ folder, do not edit it
// (it is auto-generated from Generator/*).

#include "GB.h"
#ifndef GBCUDA_DEV
#include "GB_control.h" 
#include "GB_bld__include.h"

// The reduction is defined by the following types and operators:

// Assemble tuples:    GB (_bld__eq_bool)

// S type:   bool
// T type:   bool
// X type:   bool
// Y type:   bool
// Z type:   bool

// dup:      s = (s == aij)

#define GB_S_TYPENAME \
    bool

#define GB_T_TYPENAME \
    bool

#define GB_X_TYPENAME \
    bool

#define GB_Y_TYPENAME \
    bool

#define GB_Z_TYPENAME \
    bool

// Array to array

    // W [k] = (ztype) S [i], with typecast
    #define GB_CAST_ARRAY_TO_ARRAY(W,k,S,i)         \
        W [k] = S [i]

    // W [k] += (ztype) S [i], with typecast
    #define GB_ADD_CAST_ARRAY_TO_ARRAY(W,k,S,i)     \
        W [k] = (W [k] == S [i])

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_EQ || GxB_NO_BOOL || GxB_NO_EQ_BOOL)

//------------------------------------------------------------------------------
// build a non-iso matrix
//------------------------------------------------------------------------------

GrB_Info GB (_bld__eq_bool)
(
    bool *restrict Tx,
    int64_t  *restrict Ti,
    const bool *restrict Sx,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_bld_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

#endif

