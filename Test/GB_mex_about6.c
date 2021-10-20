//------------------------------------------------------------------------------
// GB_mex_about6: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_dynamic.h"
#include "GB_serialize.h"

#define USAGE "GB_mex_about6"
#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

typedef int32_t myint ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix C = NULL, A = NULL, B = NULL, P = NULL ;
    GrB_Scalar Amissing = NULL, Bmissing = NULL ;
    GrB_Type MyType = NULL ;
    char *err ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // eWiseUnion
    //--------------------------------------------------------------------------

    OK (GrB_Type_new (&MyType, sizeof (myint))) ;

    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&B, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&C, GrB_FP64, 10, 10)) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, Amissing,
        B, Bmissing, NULL)) ;

    OK (GrB_Scalar_new (&Amissing, GrB_FP64)) ;
    OK (GrB_Scalar_new (&Bmissing, GrB_FP64)) ;

    expected = GrB_EMPTY_OBJECT ;
    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, Amissing,
        B, Bmissing, NULL)) ;
    OK (GrB_Matrix_error_(&err, C)) ;
    printf ("expected error:\n%s\n", err) ;

    OK (GrB_Scalar_setElement_FP64_ (Amissing, (double) 42)) ;
    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, Amissing,
        B, Bmissing, NULL)) ;
    OK (GrB_Matrix_error_(&err, C)) ;
    printf ("expected error:\n%s\n", err) ;

    GrB_Scalar_free_(&Amissing) ;
    OK (GrB_Scalar_new (&Amissing, MyType)) ;
    myint nothing [1] ;
    memset (nothing, 0, sizeof (myint)) ;
    OK (GrB_Scalar_setElement_UDT (Amissing, (void *) nothing)) ;
    OK (GxB_Scalar_fprint (Amissing, "Amissing", 3, NULL)) ;
    OK (GxB_Scalar_fprint (Bmissing, "Bmissing", 3, NULL)) ;
    OK (GrB_Scalar_setElement_FP64 (Bmissing, (double) 99)) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, Amissing,
        B, Bmissing, NULL)) ;
    OK (GrB_Matrix_error_(&err, C)) ;
    printf ("expected error:\n%s\n", err) ;

    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, Bmissing,
        B, Amissing, NULL)) ;
    OK (GrB_Matrix_error_(&err, C)) ;
    printf ("expected error:\n%s\n", err) ;

    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&B) ;
    GrB_Matrix_free_(&C) ;
    GrB_Scalar_free_(&Amissing) ;
    GrB_Scalar_free_(&Bmissing) ;
    GrB_Type_free_(&MyType) ;

    //--------------------------------------------------------------------------
    // sort
    //--------------------------------------------------------------------------

    expected = GrB_NULL_POINTER ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_setElement_FP64 (A, (double) 1.2, 0, 0)) ;
    ERR (GxB_Matrix_sort (NULL, NULL, GrB_LT_FP64, A, NULL)) ;

    OK (GrB_Matrix_new (&C, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&P, GrB_INT64, 10, 10)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GxB_Matrix_sort (C, P, GrB_PLUS_FP64, A, NULL)) ;

    GrB_Matrix_free_(&C) ;
    OK (GrB_Matrix_new (&C, GrB_FP64, 9, 10)) ;

    expected = GrB_DIMENSION_MISMATCH ;
    ERR (GxB_Matrix_sort (C, P, GrB_LT_FP64, A, NULL)) ;

    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&P) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;   
    printf ("\nGB_mex_about6: all tests passed\n\n") ;
}

