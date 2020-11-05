//------------------------------------------------------------------------------
// GB_mex_about2: more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_about2"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GxB_Scalar scalar = NULL ;
    GrB_Vector victor = NULL ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    FILE *f = fopen ("errlog3.txt", "w") ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // test removeElement/setElement when jumbled
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_INT32, 10, 10)) ;
    OK (GrB_Vector_new (&victor, GrB_INT32, 10)) ;
    OK (GxB_Scalar_new (&scalar, GrB_INT32)) ;

    OK (GxB_Matrix_fprint (A, "A before set", 3, NULL)) ;
    OK (GrB_Matrix_setElement_INT32 (A, 314159, 0, 0)) ;
    OK (GxB_Matrix_fprint (A, "A after set", 3, NULL)) ;
    A->jumbled = true ;
    OK (GrB_Matrix_removeElement (A, 0, 0)) ;
    OK (GxB_Matrix_fprint (A, "A after remove", 3, NULL)) ;
    A->jumbled = true ;
    OK (GrB_Matrix_setElement_INT32 (A, 99099, 0, 0)) ;
    OK (GxB_Matrix_fprint (A, "A after set again", 3, NULL)) ;

    OK (GxB_Vector_fprint (victor, "victor before set", 3, NULL)) ;
    OK (GrB_Vector_setElement_INT32 (victor, 44, 0)) ;
    OK (GxB_Vector_fprint (victor, "victor after set", 3, NULL)) ;
    victor->jumbled = true ;
    OK (GrB_Vector_removeElement (victor, 0)) ;
    OK (GxB_Vector_fprint (victor, "victor remove set", 3, NULL)) ;
    victor->jumbled = true ;
    OK (GrB_Vector_setElement_INT32 (victor, 88, 0)) ;
    OK (GxB_Vector_fprint (victor, "victor after set again", 3, NULL)) ;

    OK (GxB_Scalar_fprint (scalar, "scalar before set", 3, NULL)) ;
    OK (GxB_Scalar_setElement_INT32 (scalar, 404)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after set", 3, NULL)) ;
    int i = 0 ;
    OK (GxB_Scalar_extractElement_INT32 (&i, scalar)) ;
    CHECK (i == 404) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after extract", 3, NULL)) ;
    OK (GrB_Matrix_removeElement ((GrB_Matrix) scalar, 0, 0)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after remove", 3, NULL)) ;
    i = 777 ;
    expected = GrB_NO_VALUE ;
    ERR (GxB_Scalar_extractElement_INT32 (&i, scalar)) ;
    CHECK (i == 777) ;

    // force a zombie into the scalar
    OK (GxB_Scalar_setElement_INT32 (scalar, 707)) ;
    OK (GxB_Scalar_wait (&scalar)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after wait", 3, NULL)) ;
    OK (GxB_Matrix_Option_set (scalar, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    CHECK (scalar->i != NULL) ;
    scalar->i [0] = GB_FLIP (0) ;
    scalar->nzombies = 1 ;
    OK (GxB_Scalar_fprint (scalar, "scalar with zombie", 3, NULL)) ;
    expected = GrB_NO_VALUE ;
    ERR (GxB_Scalar_extractElement_INT32 (&i, scalar)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar after extract", 3, NULL)) ;
    CHECK (i == 777) ;

    GrB_Vector_free_(&victor) ;
    GrB_Matrix_free_(&A) ;
    GxB_Scalar_free_(&scalar) ;

    //--------------------------------------------------------------------------
    // builtin comparators not defined for complex types
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GxB_FC32, 10, 10)) ;
    OK (GrB_Matrix_new (&A, GxB_FC32, 10, 10)) ;
    OK (GxB_Scalar_new (&scalar, GxB_FC32)) ;
    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GxB_Matrix_select (C, NULL, NULL, GxB_LT_THUNK, A, scalar, NULL)) ;
    char *message = NULL ;
    OK (GrB_Matrix_error (&message, C)) ;
    printf ("error expected: %s\n", message) ;

    GrB_Matrix_free_(&C) ;
    GrB_Matrix_free_(&A) ;
    GxB_Scalar_free_(&scalar) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;   
    fclose (f) ;
    printf ("\nAll errors printed above were expected.\n") ;
    printf ("GB_mex_about2: all tests passed\n\n") ;
}

