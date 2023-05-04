//------------------------------------------------------------------------------
// GB_mex_about16: JIT error handling
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_stringify.h"

#define USAGE "GB_mex_about16"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void myfunc (float *z, const float *x) ;
void myfunc (float *z, const float *x) { (*z) = (*x) + 1 ; }

void mymult (float *z, const float *x, const float *y) ;
void mymult (float *z, const float *x, const float *y) { (*z) = (*x)*(*y) + 1 ;}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // create some valid matrices
    //--------------------------------------------------------------------------

    GrB_Index n = 4 ;
    GrB_Matrix A = NULL, B = NULL, F = NULL, C = NULL, D = NULL, G = NULL,
        S = NULL, H = NULL, F2 = NULL, F3 = NULL ;
    OK (GrB_Matrix_new (&A, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&B, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&D, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&F, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&F2, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&G, GrB_FP32, 2*n, 2*n)) ;
    OK (GrB_Matrix_new (&S, GrB_FP32, 200, 200)) ;
    OK (GrB_Matrix_new (&H, GrB_FP32, 400, 400)) ;
    OK (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GxB_set (B, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    OK (GxB_set (D, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GxB_set (H, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GxB_set (F, GxB_SPARSITY_CONTROL, GxB_FULL)) ;
    OK (GxB_set (F2, GxB_SPARSITY_CONTROL, GxB_FULL)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_assign (B, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_assign (F, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_assign (F2, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_Matrix_setElement (A, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (B, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (F, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (F2, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (S, 2, 10, 4)) ;
    for (int i = 0 ; i < 200 ; i++)
    {
        OK (GrB_Matrix_setElement (S, 3, i, i)) ;
    }
    OK (GrB_select (D, NULL, NULL, GrB_DIAG, A, 0, NULL)) ;
    OK (GrB_Matrix_dup (&F3, F2)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (B, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (F, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (F2, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (F3, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (D, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (G, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (H, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (S, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // set the JIT to ON, and the Factory Kernels off
    //--------------------------------------------------------------------------

    int save_control ;
    bool save_factory = GB_factory_kernels_enabled ;
    GB_factory_kernels_enabled = false ;
    OK (GxB_Global_Option_get_INT32 (GxB_JIT_C_CONTROL, &save_control)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    //--------------------------------------------------------------------------
    // try some methods that require the JIT
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_BURBLE, true)) ;
    GrB_Semiring s ;
    GrB_BinaryOp mult ;
    GrB_Monoid mon ;

    // user type with zero size
    GrB_Type MyType ;
    GrB_Info expected = GrB_INVALID_VALUE ;
    ERR (GxB_Type_new (&MyType, 0, NULL, NULL)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GrB_Type_new (&MyType, sizeof (double))) ;
    OK (GxB_print (MyType, 3)) ;
    size_t size ;
    expected = GrB_NO_VALUE ;
    ERR (GB_user_type_jit (&size, MyType)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    printf ("\nmacrofy type:\n") ;
    GB_macrofy_user_type (NULL, MyType) ;

    // user function with NULL pointer
    GrB_UnaryOp op ;
    expected = GrB_NULL_POINTER ;
    ERR (GxB_UnaryOp_new (&op, NULL, GrB_FP32, GrB_FP32, NULL, NULL)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    OK (GrB_UnaryOp_new (&op, myfunc, GrB_FP32, GrB_FP32)) ;
    printf ("\nmacrofy op:\n") ;
    GB_macrofy_user_op (NULL, op) ;

    OK (GrB_BinaryOp_new (&mult, mymult, GrB_FP32, GrB_FP32, GrB_FP32)) ;
    OK (GxB_print (mult, 3)) ;

    OK (GrB_Monoid_new (&mon, mult, (float) 1)) ;
    OK (GxB_print (mult, 3)) ;

    OK (GrB_Semiring_new (&s, mon, mult)) ;
    OK (GxB_print (s, 3)) ;

    GB_jit_encoding e ;
    char *suffix ;
    uint64_t code = GB_encodify_mxm (&e, &suffix, 0, false, false, GxB_SPARSE,
        GrB_FP32, NULL, false, false, s, false, A, B) ;
    CHECK (code == UINT64_MAX) ;

    code = GB_encodify_reduce (&e, &suffix, mon, A) ;
    CHECK (code == UINT64_MAX) ;

    code = GB_encodify_assign (&e, &suffix, 0, C, false, 0, 0, NULL,
        false, false, mult, A, NULL, 0) ;
    CHECK (code == UINT64_MAX) ;

    code = GB_encodify_build (&e, &suffix, 0, mult, GrB_FP32, GrB_FP32) ;
    CHECK (code == UINT64_MAX) ;

    //--------------------------------------------------------------------------
    // restore the JIT control, Factory Kernels, and renable the JIT
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_BURBLE, false)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, save_control)) ;
    GB_factory_kernels_enabled = save_factory ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GrB_free (&B) ;
    GrB_free (&C) ;
    GrB_free (&D) ;
    GrB_free (&F) ;
    GrB_free (&G) ;
    GrB_free (&H) ;
    GrB_free (&S) ;
    GrB_free (&F2) ;
    GrB_free (&F3) ;
    GrB_free (&MyType) ;
    GrB_free (&op) ;
    GrB_free (&mult) ;
    GrB_free (&s) ;
    GrB_free (&mon) ;

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_about16:  all tests passed\n\n") ;
}

