//------------------------------------------------------------------------------
// gb_get_first_scalar: x = find (V, 'first')
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_UTIL

#define FREE_WORK           \
    GrB_Vector_free (&T) ;

#define FREE_ALL            \
    FREE_WORK ;             \
    GrB_Scalar_free (x) ;

#include "gb_interface.h"

GrB_Info gb_get_first_scalar
(
    GrB_Scalar *x,          // x = find (V, 'first')
    GrB_Vector V,
    GrB_Type type
)
{ 

    //--------------------------------------------------------------------------
    // get the first entry from a vector V
    //--------------------------------------------------------------------------

//  printf ("\n------------- gb_get_first_scalar:\n") ;
//  GxB_Vector_fprint (V, "V input to gb_get_first_scalar", 5, NULL) ;
//  GxB_Type_fprint (type, "type", 5, NULL) ;

    (*x) = NULL ;
    GrB_Vector T = NULL ;

    OK (GrB_Scalar_new (x, type)) ;
//  GxB_Scalar_fprint (*x, "allocated x", 5, NULL) ;
    OK (GrB_Vector_new (&T, type, 0)) ;
    OK (GxB_Vector_extractTuples_Vector (NULL, T, V, NULL)) ;
    OK (GrB_Vector_extractElement_Scalar (*x, T, 0)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    return (GrB_SUCCESS) ;
}

