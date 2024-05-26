//------------------------------------------------------------------------------
// GB_red:  hard-coded functions for reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_control.h" 
#include "GB_red__include.h"

// reduction operator and type:
#define GB_UPDATE(z,a)  z = a
#define GB_ADD(z,zin,a) z = a
#define GB_GETA_AND_UPDATE(z,Ax,p) z = Ax [p]

// A matrix (no typecasting to Z type here)
#define GB_A_TYPE uint8_t
#define GB_DECLAREA(aij) uint8_t aij
#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [pA]

// monoid properties:
#define GB_Z_TYPE uint8_t
#define GB_DECLARE_IDENTITY(z) uint8_t z = 0
#define GB_DECLARE_IDENTITY_CONST(z) const uint8_t z = 0
#define GB_IS_ANY_MONOID 1
#define GB_MONOID_IS_TERMINAL 1

// panel size

// disable this operator and use the generic case if these conditions hold
#if (defined(GxB_NO_ANY) || defined(GxB_NO_UINT8) || defined(GxB_NO_ANY_UINT8))
#define GB_DISABLE 1
#else
#define GB_DISABLE 0
#endif

#include "GB_monoid_shared_definitions.h"

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

GrB_Info GB (_red__any_uint8)
(
    GB_Z_TYPE *result,
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
    GB_Z_TYPE z = (*result) ;
    GB_Z_TYPE *restrict W = (GB_Z_TYPE *) W_space ;
    if (A->nzombies > 0 || GB_IS_BITMAP (A))
    {
        #include "template/GB_reduce_to_scalar_template.c"
    }
    else
    {
        #include "template/GB_reduce_panel.c"
    }
    (*result) = z ;
    return (GrB_SUCCESS) ;
    #endif
}

