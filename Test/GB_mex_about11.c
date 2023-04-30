//------------------------------------------------------------------------------
// GB_mex_about11: JIT testing and set/get
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_file.h"

#define USAGE "GB_mex_about11"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void myplus (float *z, const float *x, const float *y) ;
void myplus (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }
#define MYPLUS_DEFN \
"void myplus (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }"

void myinc (float *z, const float *x) ;
void myinc (float *z, const float *x) { (*z) = (*x)+1 ; }
#define MYINC_DEFN \
"void myinc (float *z, const float *x) { (*z) = (*x)+1 ; }"

void myidx (int64_t *z, const void *x, GrB_Index i, GrB_Index j, const void *y);
void myidx (int64_t *z, const void *x, GrB_Index i, GrB_Index j, const void *y)
{
    (*z) = i + j ;
}
#define MYIDX_DEFN \
"void myidx (int64_t *z, const void *x, GrB_Index i, GrB_Index j," \
" const void *y)\n" \
"{ \n" \
"    (*z) = (int64_t) (i + j) ; \n" \
"}"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Matrix A = NULL ;
    GrB_Vector v = NULL ;
    GrB_Scalar scalar = NULL ;
    char *s = NULL, *t = NULL, *c = NULL, *cache = NULL ;
    GrB_Info info, expected ;
    bool onebased = false, iso = false ;
    int use_cmake_int = 0, onebased_int = 1, control = 99 ;
    size_t mysize = 99 ;
    bool use_cmake = false ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Descriptor desc = NULL ;
    bool malloc_debug = GB_mx_get_global (true) ;
    printf ("GB_mex_about11: malloc_debug: %d\n", malloc_debug) ;

    //--------------------------------------------------------------------------
    // remove temp files and folders
    //--------------------------------------------------------------------------

    remove ("/tmp/grberr2.txt") ;
    remove ("/tmp/grb_error_log.txt") ;
    system ("cmake -E remove_directory /tmp/grb_cache") ;

    //--------------------------------------------------------------------------
    // get/set tests
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GrB_Matrix_new (&A, GrB_FP32, 3, 4)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, 3, GrB_ALL, 4, NULL)) ;
    OK (GxB_Matrix_iso (&iso, A)) ;
    CHECK (iso) ;
    OK (GrB_Matrix_setElement (A, 3, 0, 0)) ;
    OK (GxB_Matrix_iso (&iso, A)) ;
    CHECK (!iso) ;

    OK (GrB_Vector_new (&v, GrB_FP32, 3)) ;
    OK (GrB_assign (v, NULL, NULL, 1, GrB_ALL, 3, NULL)) ;
    OK (GxB_Vector_iso (&iso, v)) ;
    CHECK (iso) ;
    OK (GrB_Vector_setElement (v, 3, 0)) ;
    OK (GxB_Vector_iso (&iso, v)) ;
    CHECK (!iso) ;
    OK (GrB_free (&v)) ;

    OK (GxB_set (GxB_PRINT_1BASED, true)) ;
    OK (GxB_print (A, 3)) ;
    OK (GxB_get (GxB_PRINT_1BASED, &onebased)) ;
    CHECK (onebased == true) ;

    OK (GxB_Global_Option_set_INT32 (GxB_PRINT_1BASED, false)) ;
    OK (GxB_print (A, 3)) ;
    OK (GxB_Global_Option_get_INT32 (GxB_PRINT_1BASED, &onebased_int)) ;
    CHECK (onebased_int == 0) ;
    OK (GrB_free (&A)) ;

    OK (GxB_set (GxB_BURBLE, true)) ;
    OK (GxB_set (GxB_JIT_USE_CMAKE, true)) ;
    OK (GxB_get (GxB_JIT_USE_CMAKE, &use_cmake)) ;
    CHECK (use_cmake == true) ;
    OK (GxB_Global_Option_get_INT32 (GxB_JIT_USE_CMAKE, &use_cmake_int)) ;
    CHECK (use_cmake_int == 1) ;

    GrB_Type MyType = NULL ;
    OK (GxB_Type_new (&MyType, 0, "mytype", "typedef double mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (double)) ;
    OK (GrB_free (&MyType)) ;

    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_NAME, &c)) ;
    printf ("default compiler [%s]\n", c) ;
    OK (GxB_set (GxB_JIT_C_COMPILER_NAME, "cc")) ;
    OK (GxB_get (GxB_JIT_C_COMPILER_NAME, &s)) ;
    CHECK (MATCH (s, "cc")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_NAME, &t)) ;
    CHECK (MATCH (t, "cc")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_NAME, "gcc")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_NAME, &t)) ;
    CHECK (MATCH (t, "gcc")) ;

    #ifdef __APPLE__
    // reset the compiler back to the default on the Mac
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_NAME, c)) ;
    #endif

    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_FLAGS, &s)) ;
    printf ("default flags [%s]\n", s) ;
    OK (GxB_set (GxB_JIT_C_COMPILER_FLAGS, "-g")) ;
    OK (GxB_get (GxB_JIT_C_COMPILER_FLAGS, &s)) ;
    CHECK (MATCH (s, "-g")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_COMPILER_FLAGS, &t)) ;
    CHECK (MATCH (t, "-g")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_COMPILER_FLAGS, "-O0")) ;
    CHECK (MATCH (t, "-O0")) ;

    OK (GxB_get (GxB_JIT_C_CMAKE_LIBS, &s)) ;
    printf ("default C cmake libs [%s]\n", s) ;
    printf ("set cmake libs:\n") ;
    OK (GxB_set (GxB_JIT_C_CMAKE_LIBS, "m")) ;
    printf ("get cmake libs:\n") ;
    OK (GxB_get (GxB_JIT_C_CMAKE_LIBS, &s)) ;
    CHECK (MATCH (s, "m")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_CMAKE_LIBS, &t)) ;
    CHECK (MATCH (t, "m")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_CMAKE_LIBS, "m;dl")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_CMAKE_LIBS, &t)) ;
    CHECK (MATCH (t, "m;dl")) ;

    OK (GxB_Type_new (&MyType, 0, "mytype", "typedef int32_t mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (int32_t)) ;
    OK (GrB_free (&MyType)) ;

    OK (GxB_Global_Option_set_INT32 (GxB_JIT_USE_CMAKE, false)) ;
    OK (GxB_get (GxB_JIT_USE_CMAKE, &use_cmake)) ;
    CHECK (use_cmake == false) ;

    OK (GxB_get (GxB_JIT_C_LINKER_FLAGS, &s)) ;
    printf ("default linker flags [%s]\n", s) ;
    OK (GxB_set (GxB_JIT_C_LINKER_FLAGS, "-shared")) ;
    OK (GxB_get (GxB_JIT_C_LINKER_FLAGS, &s)) ;
    CHECK (MATCH (s, "-shared")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LINKER_FLAGS, &t)) ;
    CHECK (MATCH (t, "-shared")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_LINKER_FLAGS, " -shared  ")) ;
    CHECK (MATCH (t, " -shared  ")) ;

    OK (GxB_get (GxB_JIT_C_LIBRARIES, &s)) ;
    printf ("default C libraries [%s]\n", s) ;
    OK (GxB_set (GxB_JIT_C_LIBRARIES, "-lm")) ;
    OK (GxB_get (GxB_JIT_C_LIBRARIES, &s)) ;
    CHECK (MATCH (s, "-lm")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LIBRARIES, &t)) ;
    CHECK (MATCH (t, "-lm")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_LIBRARIES, "-lm -ldl")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_LIBRARIES, &t)) ;
    CHECK (MATCH (t, "-lm -ldl")) ;

    OK (GxB_get (GxB_JIT_C_PREFACE, &s)) ;
    printf ("default C preface [%s]\n", s) ;
    OK (GxB_set (GxB_JIT_C_PREFACE, "// stuff here")) ;
    OK (GxB_get (GxB_JIT_C_PREFACE, &s)) ;
    CHECK (MATCH (s, "// stuff here")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_PREFACE, &t)) ;
    CHECK (MATCH (t, "// stuff here")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_C_PREFACE, "// more stuff here")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_C_PREFACE, &t)) ;
    CHECK (MATCH (t, "// more stuff here")) ;

    OK (GxB_Type_new (&MyType, 0, "mytype", "typedef double mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (double)) ;
    OK (GrB_free (&MyType)) ;

    OK (GxB_Type_new (&MyType, 0, "mytype", "typedef int32_t mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (int32_t)) ;
    OK (GrB_free (&MyType)) ;

    printf ("\n--------------------------- intentional compile errors:\n") ;
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Type_new (&MyType, 0, "mytype2", "garbage")) ;
    CHECK (MyType == NULL) ;
    printf ("\n-------------------------------------------------------\n\n") ;

    OK (GxB_get (GxB_JIT_C_CONTROL, &control)) ;
    CHECK (control == GxB_JIT_LOAD) ;
    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GxB_get (GxB_JIT_C_CONTROL, &control)) ;
    CHECK (control == GxB_JIT_ON) ;

    OK (GxB_get (GxB_JIT_ERROR_LOG, &s)) ;
    printf ("default error log: [%s]\n", s) ;
    OK (GxB_set (GxB_JIT_ERROR_LOG, "/tmp/grb_error_log.txt")) ;
    OK (GxB_get (GxB_JIT_ERROR_LOG, &t)) ;
    printf ("new error log: [%s]\n", t) ;
    CHECK (MATCH (t, "/tmp/grb_error_log.txt")) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Type_new (&MyType, 0, "mytype2", "garbage")) ;
    CHECK (MyType == NULL) ;

    printf ("\n------------------------ compile error log (intentional):\n") ;
    system ("cat /tmp/grb_error_log.txt") ;
    printf ("\n-------------------------------------------------------\n\n") ;

    OK (GxB_Global_Option_get_CHAR (GxB_JIT_ERROR_LOG, &s)) ;
    CHECK (MATCH (s, "/tmp/grb_error_log.txt")) ;
    OK (GxB_Global_Option_set_CHAR (GxB_JIT_ERROR_LOG, "/tmp/grberr2.txt")) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_ERROR_LOG, &s)) ;
    CHECK (MATCH (s, "/tmp/grberr2.txt")) ;

    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Type_new (&MyType, 0, "mytype2", "more garbage")) ;
    CHECK (MyType == NULL) ;

    printf ("\n------------------------ compile error log (intentional):\n") ;
    system ("cat /tmp/grberr2.txt") ;
    printf ("\n-------------------------------------------------------\n\n") ;

    OK (GxB_get (GxB_JIT_CACHE_PATH, &cache)) ;
    printf ("default cache path: [%s]\n", cache) ;
    OK (GxB_set (GxB_JIT_CACHE_PATH, "/tmp/grb_cache")) ;
    OK (GxB_get (GxB_JIT_CACHE_PATH, &s)) ;
    printf ("new cache path: [%s]\n", s) ;
    CHECK (MATCH (s, "/tmp/grb_cache")) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Global_Option_set_CHAR (999, "gunk")) ;

    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_OFF)) ;
    OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GxB_Type_new (&MyType, 0, "mytype", "typedef double mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (double)) ;
    OK (GrB_free (&MyType)) ;


    OK (GxB_Type_new (&MyType, 0, "mytype", "typedef int32_t mytype ;")) ;
    OK (GxB_Type_size (&mysize, MyType)) ;
    CHECK (mysize == sizeof (int32_t)) ;

    expected = GrB_INVALID_OBJECT ;
    MyType->name_len++ ;
    ERR (GxB_print (MyType, 3)) ;
    MyType->name_len-- ;
    OK (GxB_print (MyType, 3)) ;
    OK (GrB_free (&MyType)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Type_new (&MyType, 0, NULL, NULL)) ;
    ERR (GxB_Type_new (&MyType, 0, NULL, "typedef int32_t mytype ;")) ;
    ERR (GxB_Type_new (&MyType, 0, "mytype", NULL)) ;

    OK (GxB_Global_Option_set_CHAR (GxB_JIT_CACHE_PATH, cache)) ;
    OK (GxB_Global_Option_get_CHAR (GxB_JIT_CACHE_PATH, &s)) ;
    CHECK (MATCH (s, cache)) ;

    //--------------------------------------------------------------------------
    // GrB_Semiring_new memory tests
    //--------------------------------------------------------------------------

    GrB_Semiring sr = NULL ;
    GrB_BinaryOp op = NULL ;
    GrB_Monoid monoid = NULL ;
    METHOD (GrB_Semiring_new (&sr, GrB_PLUS_MONOID_FP32, GrB_TIMES_FP32)) ;
    OK (GxB_print (sr, 3)) ;
    GrB_free (&sr) ;

    OK (GxB_BinaryOp_new (&op, myplus, GrB_FP32, GrB_FP32, GrB_FP32, "myplus",
        MYPLUS_DEFN)) ;
    float zero = 0 ;
    OK (GrB_Monoid_new (&monoid, op, zero)) ;
    OK (GxB_print (op, 3)) ;
    OK (GxB_print (monoid, 3)) ;

    OK (GrB_Semiring_new (&sr, monoid, op)) ;
    OK (GxB_print (sr, 3)) ;

    expected = GrB_INVALID_OBJECT ;
    op->name_len-- ;
    ERR (GxB_print (op, 3)) ;
    op->name_len++ ;
    OK (GxB_print (op, 3)) ;

    sr->name_len-- ;
    ERR (GxB_print (sr, 3)) ;
    sr->name_len++ ;
    OK (GxB_print (sr, 3)) ;

    GrB_free (&sr) ;
    GrB_free (&monoid) ;
    GrB_free (&op) ;

    //--------------------------------------------------------------------------
    // GxB_select (deprecated but still functional)
    //--------------------------------------------------------------------------

    GrB_Type type = GrB_INT64 ;
    OK (GxB_SelectOp_ttype (&type, GxB_TRIL)) ;
    CHECK (type == NULL) ;
    type = GrB_INT64 ;
    OK (GxB_SelectOp_xtype (&type, GxB_TRIL)) ;
    CHECK (type == NULL) ;
    expected = GrB_NULL_POINTER ;
    ERR (GxB_SelectOp_fprint (NULL, "null", 3, stdout)) ;

    OK (GrB_Matrix_new (&A, GxB_FC32, 3, 4)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, 3, GrB_ALL, 4, NULL)) ;
    OK (GrB_Matrix_setElement (A, 2, 0, 0)) ;
    OK (GrB_Matrix_setElement (A, 3, 0, 1)) ;
    OK (GrB_Scalar_new (&scalar, GxB_FC32)) ;
    OK (GrB_Scalar_setElement (scalar, (float) 2)) ;
    OK (GxB_Matrix_select (A, NULL, NULL, GxB_NE_THUNK, A, scalar, NULL)) ;
    OK (GxB_print (A, 3)) ;
    OK (GrB_Scalar_setElement (scalar, (float) 1)) ;
    OK (GxB_Matrix_select (A, NULL, NULL, GxB_EQ_THUNK, A, scalar, NULL)) ;
    OK (GxB_print (A, 3)) ;
    GrB_free (&A) ;
    GrB_free (&scalar) ;

    //--------------------------------------------------------------------------
    // GxB_print, GrB_IndexUnaryOp, and GrB_UnaryOp
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GxB_FC32, 30, 30)) ;
    OK (GrB_Matrix_setElement (A, 2, 4, 5)) ;
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    expected = GrB_INVALID_OBJECT ;
    A->nvals++ ;
    ERR (GxB_print (A, 3)) ;
    scalar = A ;
    ERR (GxB_print (scalar, 3)) ;
    A->nvals-- ;
    OK (GxB_print (A, 3)) ;

    GrB_IndexUnaryOp MyIdxOp = NULL ;
    #undef GrB_IndexUnaryOp_new
    #undef GrM_IndexUnaryOp_new
    OK (GrM_IndexUnaryOp_new (&MyIdxOp, myidx,
        GrB_INT64, GrB_INT64, GrB_INT64)) ;
    OK (GxB_print (MyIdxOp, 3)) ;
    OK (GrB_apply (A, NULL, NULL, MyIdxOp, A, 0, NULL)) ;
    OK (GxB_print (A, 3)) ;

    MyIdxOp->name_len++ ;
    ERR (GxB_print (MyIdxOp, 3)) ;
    MyIdxOp->name_len-- ;
    OK (GxB_print (MyIdxOp, 3)) ;
    GrB_free (&MyIdxOp) ;

    GrB_UnaryOp MyUnOp = NULL ;
    OK (GrB_UnaryOp_new (&MyUnOp, myinc, GrB_FP32, GrB_FP32)) ;
    OK (GxB_print (MyUnOp, 3)) ;
    OK (GrB_apply (A, NULL, NULL, MyUnOp, A, NULL)) ;
    OK (GxB_print (A, 3)) ;

    MyUnOp->name_len++ ;
    ERR (GxB_print (MyUnOp, 3)) ;
    MyUnOp->name_len-- ;
    OK (GxB_print (MyUnOp, 3)) ;
    GrB_free (&MyUnOp) ;

    GrB_free (&MyUnOp) ;
    GrB_free (&A) ;

    //--------------------------------------------------------------------------
    // GB_file tests
    //--------------------------------------------------------------------------

    bool ok = GB_file_mkdir (NULL) ;
    CHECK (!ok) ;
    ok = GB_file_unlock_and_close (NULL, NULL) ;
    CHECK (!ok) ;
    FILE *fp = NULL ;
    int fd = -1 ;
    ok = GB_file_unlock_and_close (&fp, &fd) ;
    CHECK (!ok) ;

    ok = GB_file_open_and_lock (NULL, NULL, NULL) ;
    CHECK (!ok) ;

    ok = GB_file_open_and_lock ("/nopermission", &fp, &fd) ;
    CHECK (!ok) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    // remove temp files and folders
    remove ("/tmp/grberr2.txt") ;
    remove ("/tmp/grb_error_log.txt") ;
    system ("cmake -E remove_directory /tmp/grb_cache") ;

    OK (GxB_set (GxB_BURBLE, false)) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_about11: all tests passed\n\n") ;
}

