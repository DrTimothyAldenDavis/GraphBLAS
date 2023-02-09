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

// Reduce to scalar:   GB (_red__min_uint64)

// A type:   uint64_t
// Z type:   uint64_t

// Update:   if (y < z) { z = y ; }
// Add func: z = GB_IMIN (x, y)

#define GB_A_TYPE \
    uint64_t

#define GB_Z_TYPE \
    uint64_t

// declare a scalar and set it equal to the monoid identity value
#define GB_DECLARE_IDENTITY(z) uint64_t z = UINT64_MAX
#define GB_DECLARE_IDENTITY_CONST(z) const uint64_t z = UINT64_MAX

// reduction operator:

    // declare aij as ztype (= atype since no typecasting is done here)
    #define GB_DECLAREA(aij)  \
        uint64_t aij

    // aij = Ax [pA]
    #define GB_GETA(aij,Ax,pA,A_iso)  \
        aij = Ax [pA]

    // z += y, update
    #define GB_UPDATE(z,y) \
        if (y < z) { z = y ; }

    // z = x+y, additive function
    #define GB_ADD(z,x,y) \
        z = GB_IMIN (x, y)

    // s += (ztype) Ax [p], no typecast here however
    #define GB_GETA_AND_UPDATE(s,Ax,p)              \
        GB_UPDATE (s, Ax [p])

// monoid terminal condition, if any:

#define GB_MONOID_IS_TERMINAL 1
#define GB_TERMINAL_CONDITION(z,zterminal) (z == 0)
#define GB_IF_TERMINAL_BREAK(z,zterminal) if (z == 0) { break ; }
#define GB_DECLARE_TERMINAL_CONST(zterminal) const uint64_t zterminal = 0

// panel size for built-in operators

    #define GB_PANEL                                \
        16

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_MIN || GxB_NO_UINT64 || GxB_NO_MIN_UINT64)

#include "GB_monoid_shared_definitions.h"

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

GrB_Info GB (_red__min_uint64)
(
    uint64_t *result,
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
    uint64_t z = (*result) ;
    uint64_t *restrict W = (uint64_t *) W_space ;
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

