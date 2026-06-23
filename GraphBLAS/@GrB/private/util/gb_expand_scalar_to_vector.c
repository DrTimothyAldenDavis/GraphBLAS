//------------------------------------------------------------------------------
// gb_expand_scalar_to_vector: V (1:nvals) = W (1st entry)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_UTIL

#define FREE_WORK               \
    GrB_Scalar_free (&x) ;

#define FREE_ALL                \
    FREE_WORK                   \
    GrB_Vector_free (V) ;

#include "gb_interface.h"

GrB_Info gb_expand_scalar_to_vector
(
    // output
    GrB_Vector *V,
    // input
    GrB_Vector W,
    GrB_Type type,
    uint64_t nvals,
    const int arena,
    char err [ERRLEN]
)
{ 

    //--------------------------------------------------------------------------
    // get the single entry from the input vector V
    //--------------------------------------------------------------------------

    GrB_Scalar x = NULL ;
    OK (gb_get_first_scalar (&x, W, type, arena, err)) ;

    //--------------------------------------------------------------------------
    // expand the scalar into V, of length nvals
    //--------------------------------------------------------------------------

    OK (GxB_Vector_new_arena (V, type, nvals, arena, arena)) ;
    OK (GxB_Vector_assign_Scalar_Vector (*V, NULL, NULL, x, NULL, NULL)) ;
    FREE_WORK ;
    return (GrB_SUCCESS) ;
}

