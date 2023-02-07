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

// Reduce to scalar:   GB (_red)

// A type:   GB_atype
// Z type:   GB_ztype

// Update:   GB_update_op(z,y)
// Add func: GB_add_op(z,x,y)
// Identity: GB_identity
// Terminal: GB_if_terminal_break(z)

#define GB_A_TYPE \
    GB_atype

#define GB_Z_TYPE \
    GB_ztype

// declare a scalar and set it equal to the monoid identity value

    #define GB_DECLARE_MONOID_IDENTITY(modifier,z) \
        modifier GB_ztype z = GB_identity

// reduction operator:

    // declare aij as ztype (= atype since no typecasting is done here)
    #define GB_DECLAREA(aij)  \
        GB_declarea(aij)

    // aij = Ax [pA]
    #define GB_GETA(aij,Ax,pA,A_iso)  \
        GB_geta(aij,Ax,pA,false)

    // z += y, update
    #define GB_UPDATE(z,y) \
        GB_update_op(z, y)

    // z = x+y, additive function
    #define GB_ADD(z,x,y) \
        GB_add_op(z, x, y)

    // s += (ztype) Ax [p], no typecast here however
    #define GB_GETA_AND_UPDATE(s,Ax,p)              \
        GB_UPDATE (s, Ax [p])

// monoid terminal condition, if any:
GB_is_any_monoid
GB_monoid_is_terminal
GB_terminal_condition
GB_if_terminal_break
GB_declare_monoid_terminal

// panel size for built-in operators

    #define GB_PANEL                                \
        GB_panel

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    GB_disable

#include "GB_monoid_shared_definitions.h"

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

GrB_Info GB (_red)
(
    GB_ztype *result,
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
    GB_ztype z = (*result) ;
    GB_ztype *restrict W = (GB_ztype *) W_space ;
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

