

//------------------------------------------------------------------------------
// GB_red:  hard-coded functions for reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated2/ folder, do not edit it
// (it is auto-generated from Generator/*).

#include "GB.h"
#ifndef GBCUDA_DEV
#include "GB_control.h" 
#include "GB_red__include.h"

// The reduction is defined by the following types and operators:

// Reduce to scalar:   GB (_red__min_uint8)

// A type:   uint8_t
// Z type:   uint8_t

// Reduce:   if (aij < z) { z = aij ; }
// Identity: UINT8_MAX
// Terminal: if (z == 0) { break ; }

#define GB_A_TYPENAME \
    uint8_t

#define GB_Z_TYPENAME \
    uint8_t

// declare a scalar and set it equal to the monoid identity value

    #define GB_DECLARE_MONOID_IDENTITY(z)           \
        uint8_t z = UINT8_MAX

// reduction operator:

    // z += y, no typecast
    #define GB_UPDATE(z,y) \
        if (y < z) { z = y ; }

    // s += (ztype) Ax [p], no typecast here however
    #define GB_GETA_AND_UPDATE(s,Ax,p)              \
        GB_UPDATE (s, Ax [p])

// break the loop if terminal condition reached

    #define GB_MONOID_IS_TERMINAL                   \
        1

    #define GB_TERMINAL_CONDITION(z,zterminal)      \
        (z == 0)

    #define GB_IF_TERMINAL_BREAK(z,zterminal)       \
        if (z == 0) { break ; }

// panel size for built-in operators

    #define GB_PANEL                                \
        16

// special case for the ANY monoid

    #define GB_IS_ANY_MONOID                        \
        0

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_MIN || GxB_NO_UINT8 || GxB_NO_MIN_UINT8)

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

GrB_Info GB (_red__min_uint8)
(
    uint8_t *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    uint8_t z = (*result) ;
    uint8_t *restrict W = (uint8_t *) W_space ;
    if (A->nzombies > 0 || GB_IS_BITMAP (A))
    {
        #include "GB_reduce_to_scalar_template.c"
    }
    else
    {
        #include "GB_reduce_panel.c"
    }
    (*result) = z ;
    return (GrB_SUCCESS) ;
    #endif
}

#endif

