//------------------------------------------------------------------------------
// gb_expand_scalar_to_vector: V (1:nvals) = V (1)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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
    GrB_Vector *V,
    GrB_Type type,
    uint64_t nvals
)
{ 

    //--------------------------------------------------------------------------
    // get the single entry from the input vector V, and then free it
    //--------------------------------------------------------------------------

//  GxB_Vector_fprint (*V, "V input to gb_expand_scalar_to_vector", 5, NULL) ;

    GrB_Scalar x = NULL ;
    OK (gb_get_first_scalar (&x, *V, type)) ;
//  GxB_Scalar_fprint (x, "x", 5, NULL) ;
    GrB_Vector_free (V) ;

    //--------------------------------------------------------------------------
    // expand the scalar back into V, expanding V to length nvals
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (V, type, nvals)) ;
    OK (GxB_Vector_assign_Scalar_Vector (*V, NULL, NULL, x, NULL, NULL)) ;
    GrB_Scalar_free (&x) ;
    return (GrB_SUCCESS) ;
}

