

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

// Reduce to scalar:   GB (_red__any_int64)

// A type:   int64_t
// Z type:   int64_t

// Reduce:   z = aij
// Identity: 0
// Terminal: break ;

#define GB_A_TYPENAME \
    int64_t

#define GB_Z_TYPENAME \
    int64_t

// declare a scalar and set it equal to the monoid identity value

    #define GB_DECLARE_MONOID_IDENTITY(z)           \
        int64_t z = 0

// Array to array

    // W [k] += Ax [p], no typecast
    // used in GB_reduce_panel where W == Panel
    // Panel has ztype, Ax has atype.  In general would need to
    // typecast with a temporary value, aij = (ztype) Ax [p]
    // then W [k] += aij.
    #define GB_ADD_ARRAY_TO_ARRAY(W,k,Ax,p)         \
        W [k] = Ax [p]  

// Array to scalar

    // s += (ztype) Ax [p], no typecast here
    #define GB_GETA_AND_UPDATE(s,Ax,p)     \
        s = Ax [p]

    // s += S [i], no typecast
    #define GB_ADD_ARRAY_TO_SCALAR(s,S,i)           \
        s = S [i]

// Scalar to array

    // W [k] = s, no typecast
    #define GB_COPY_SCALAR_TO_ARRAY(W,k,s)          \
        W [k] = s

// break the loop if terminal condition reached

    #define GB_MONOID_IS_TERMINAL                   \
        1

    #define GB_TERMINAL_CONDITION(z,zterminal)      \
        true

    #define GB_IF_TERMINAL_BREAK(z,zterminal)       \
        break ;

// panel size for built-in operators

    #define GB_PANEL                                \
        (no panel)

// special case for the ANY monoid

    #define GB_IS_ANY_MONOID                        \
        1

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_ANY || GxB_NO_INT64 || GxB_NO_ANY_INT64)

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

GrB_Info GB (_red__any_int64)
(
    int64_t *result,
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
    int64_t z = (*result) ;
    int64_t *restrict W = (int64_t *) W_space ;
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

