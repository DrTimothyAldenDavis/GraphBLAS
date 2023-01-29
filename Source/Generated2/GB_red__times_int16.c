

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

// Reduce to scalar:   GB (_red__times_int16)

// A type:   int16_t
// Z type:   int16_t

// Reduce:   z *= aij
// Identity: 1
// Terminal: if (z == 0) { break ; }

#define GB_A_TYPENAME \
    int16_t

#define GB_Z_TYPENAME \
    int16_t

// declare a scalar and set it equal to the monoid identity value

    #define GB_DECLARE_MONOID_IDENTITY(z)           \
        int16_t z = 1

// reduction operator:

    // z += y, no typecast
    #define GB_UPDATE(z,y) \
        z *= y

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
        64

// special case for the ANY monoid

    #define GB_IS_ANY_MONOID                        \
        0

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_TIMES || GxB_NO_INT16 || GxB_NO_TIMES_INT16)

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

GrB_Info GB (_red__times_int16)
(
    int16_t *result,
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
    int16_t z = (*result) ;
    int16_t *restrict W = (int16_t *) W_space ;
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

