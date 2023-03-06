//------------------------------------------------------------------------------
// GB_jit_kernel_apply_unop.c: Cx = op (A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_ewise_shared_definitions.h"

// cij = op (aij)
#define GB_APPLY_OP(pC,pA)                  \
{                                           \
    /* aij = Ax [pA] */                     \
    GB_DECLAREA (aij) ;                     \
    GB_GETA (aij, Ax, pA, false) ;          \
    /* Cx [pC] = unop (aij) */              \
    GB_UNARYOP (Cx [pC], aij, i, j, y) ;    \
}

GrB_Info GB_jit_kernel
(
    GB_void *Cx_out,            // Cx and Ax may be aliased
    const GB_void *Ax_in,       // A is always non-iso for this kernel
    const int8_t *restrict Ab,  // A->b if A is bitmap
    int64_t anz,
    int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GB_void *Cx_out,            // Cx and Ax may be aliased
    const GB_void *Ax_in,       // A is always non-iso for this kernel
    const int8_t *restrict Ab,  // A->b if A is bitmap
    int64_t anz,
    int nthreads
)
{ 

    GB_C_TYPE *Cx = (GB_C_TYPE *) Cx_out ;
    GB_A_TYPE *Ax = (GB_A_TYPE *) Ax_in ;
    int64_t p ;
    #if GB_A_IS_BITMAP
    { 
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < anz ; p++)
        {
            if (!Ab [p]) continue ;
            // Cx [p] = unop (Ax [p])
            GB_APPLY_OP (p, p) ;
        }
    }
    #else
    { 
        // bitmap case, no transpose; A->b already memcpy'd into C->b
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < anz ; p++)
        {
            // Cx [p] = unop (Ax [p])
            GB_APPLY_OP (p, p) ;
        }
    }
    #endif
    return (GrB_SUCCESS) ;
}

