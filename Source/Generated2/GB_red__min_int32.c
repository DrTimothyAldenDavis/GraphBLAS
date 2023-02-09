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

// Reduce to scalar:   GB (_red__min_int32)

// A type:   int32_t
// Z type:   int32_t

// Update:   if (y < z) { z = y ; }
// Add func: z = GB_IMIN (x, y)

#define GB_A_TYPE \
    int32_t

#define GB_Z_TYPE \
    int32_t

// declare a scalar and set it equal to the monoid identity value
#define GB_DECLARE_IDENTITY(z) int32_t z = INT32_MAX
#define GB_DECLARE_IDENTITY_CONST(z) const int32_t z = INT32_MAX

// reduction operator:

    // declare aij as ztype (= atype since no typecasting is done here)
    #define GB_DECLAREA(aij)  \
        int32_t aij

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
#define GB_TERMINAL_CONDITION(z,zterminal) (z == INT32_MIN)
#define GB_IF_TERMINAL_BREAK(z,zterminal) if (z == INT32_MIN) { break ; }
#define GB_DECLARE_TERMINAL_CONST(zterminal) const int32_t zterminal = INT32_MIN

// panel size for built-in operators

    #define GB_PANEL                                \
        16

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_MIN || GxB_NO_INT32 || GxB_NO_MIN_INT32)

#include "GB_monoid_shared_definitions.h"

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

GrB_Info GB (_red__min_int32)
(
    int32_t *result,
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
    int32_t z = (*result) ;
    int32_t *restrict W = (int32_t *) W_space ;
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

