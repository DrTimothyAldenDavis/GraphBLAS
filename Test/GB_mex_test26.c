//------------------------------------------------------------------------------
// GB_mex_test25: test GrB_get and GrB_set
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test26"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

typedef struct { int32_t stuff ; } mytype ;
#define MYTYPE_DEFN \
"typedef struct { int32_t stuff ; } mytype ;"

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

    GrB_Info info, expected ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Vector v = NULL ;
    GrB_Scalar s = NULL, s_fp64 = NULL, s_int32 = NULL, s_fp32 = NULL ;
    GrB_UnaryOp unop = NULL ;
    GrB_BinaryOp binop = NULL ;
    GrB_IndexUnaryOp idxunop = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Monoid monoid = NULL ;
    GrB_Type type = NULL ;
    GrB_Semiring semiring = NULL ;
    GxB_Context Context = NULL ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    char name [256] ;
    char defn [2048] ;
    int code, i ;
    float fvalue ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;

    //--------------------------------------------------------------------------
    // GrB_Type get/set
    //--------------------------------------------------------------------------

    // type name size
    OK (GrB_Type_get_SIZE_(GrB_BOOL, &size, GrB_NAME)) ;
    CHECK (size == GxB_MAX_NAME_LEN) ;

    // type name
    OK (GrB_Type_get_String_(GrB_BOOL, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_BOOL")) ;

    OK (GrB_Type_get_String_(GrB_INT8, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_INT8")) ;

    OK (GrB_Type_get_String_(GrB_INT16, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_INT16")) ;

    OK (GrB_Type_get_String_(GrB_INT32, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_INT32")) ;

    OK (GrB_Type_get_String_(GrB_INT64, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_INT64")) ;

    OK (GrB_Type_get_String_(GrB_UINT8, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_UINT8")) ;

    OK (GrB_Type_get_String_(GrB_UINT16, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_UINT16")) ;

    OK (GrB_Type_get_String_(GrB_UINT32, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_UINT32")) ;

    OK (GrB_Type_get_String_(GrB_UINT64, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_UINT64")) ;

    OK (GrB_Type_get_String_(GrB_FP32, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Type_get_String_(GrB_FP64, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_FP64")) ;

    OK (GrB_Type_get_String_(GxB_FC32, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GxB_FC32")) ;

    OK (GrB_Type_get_String_(GxB_FC64, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GxB_FC64")) ;

    // type code
    OK (GrB_Type_get_ENUM_(GrB_BOOL, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_INT8, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_INT8_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_INT16, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_INT16_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_INT32, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_INT32_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_INT64, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_INT64_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_UINT8, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_UINT8_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_UINT16, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_UINT16_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_UINT32, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_UINT32_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_UINT64, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_UINT64_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_FP32, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_Type_get_ENUM_(GrB_FP64, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_FP64_CODE) ;

    OK (GrB_Type_get_ENUM_(GxB_FC32, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GxB_FC32_CODE) ;

    OK (GrB_Type_get_ENUM_(GxB_FC64, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GxB_FC64_CODE) ;

    // type size
    OK (GrB_Type_get_SIZE_(GrB_BOOL, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (bool)) ;

    OK (GrB_Type_get_SIZE_(GrB_INT8, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (int8_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_INT16, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (int16_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_INT32, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (int32_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_INT64, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (int64_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_UINT8, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (uint8_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_UINT16, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (uint16_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_UINT32, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (uint32_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_UINT64, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (uint64_t)) ;

    OK (GrB_Type_get_SIZE_(GrB_FP32, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (float)) ;

    OK (GrB_Type_get_SIZE_(GrB_FP64, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (double)) ;

    OK (GrB_Type_get_SIZE_(GxB_FC32, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (float complex)) ;

    OK (GrB_Type_get_SIZE_(GxB_FC64, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (double complex)) ;

    // built-in type definition
    OK (GrB_Type_get_SIZE_(GrB_BOOL, &size, GxB_DEFINITION)) ;
    CHECK (size == 1) ;
    OK (GrB_Type_get_String_(GrB_BOOL, defn, GxB_DEFINITION)) ;
    CHECK (MATCH (defn, "")) ;

    // user-defined type
    #undef GrB_Type_new
    #undef GrM_Type_new
    OK (GrM_Type_new (&type, sizeof (mytype))) ;
    OK (GxB_print (type, 3)) ;
    OK (GrB_Type_set_String_(type, "mytype", GrB_NAME)) ;
    CHECK (type->hash == UINT64_MAX) ;
    OK (GrB_Type_set_String_(type, MYTYPE_DEFN, GxB_DEFINITION)) ;
    OK (GxB_print (type, 3)) ;
    CHECK (type->hash != UINT64_MAX) ;
    printf ("    hash: %016lx\n", type->hash) ;

    OK (GrB_Type_get_SIZE_(type, &size, GrB_NAME)) ;
    CHECK (size == GxB_MAX_NAME_LEN) ;
    OK (GrB_Type_get_String_(type, name, GrB_NAME)) ;
    CHECK (MATCH (name, "mytype")) ;

    OK (GrB_Type_get_SIZE_(type, &size, GxB_DEFINITION)) ;
    CHECK (size == strlen (MYTYPE_DEFN) + 1) ;
    OK (GrB_Type_get_String_(type, defn, GxB_DEFINITION)) ;
    CHECK (MATCH (defn, MYTYPE_DEFN)) ;

    OK (GrB_Type_get_SIZE_(type, &size, GrB_SIZE)) ;
    CHECK (size == sizeof (mytype)) ;

    OK (GrB_Type_get_ENUM_(type, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_UDT_CODE) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Type_get_String_(type, name, GrB_ELTYPE_STRING)) ;
    ERR (GrB_Type_get_ENUM_(type, &code, GrB_ELTYPE_STRING)) ;

    i = -1 ;
    OK (GrB_Type_get_Scalar_(type, s_int32, GrB_ELTYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GrB_UDT_CODE) ;

    OK (GrB_Type_get_Scalar_(type, s_int32, GrB_SIZE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == sizeof (mytype)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Type_get_Scalar_(type, s_int32, GrB_OUTP)) ;
    ERR (GrB_Type_get_String_(type, name, GrB_OUTP)) ;
    ERR (GrB_Type_get_SIZE_(type, &size, GrB_OUTP)) ;

    expected = GrB_NOT_IMPLEMENTED ;
    ERR (GrB_Type_get_VOID_(type, nothing, 0)) ;
    ERR (GrB_Type_set_Scalar_(type, s_int32, 0)) ;
    ERR (GrB_Type_set_ENUM_(type, 3, 0)) ;
    ERR (GrB_Type_set_VOID_(type, nothing, 0, 256)) ;

    //--------------------------------------------------------------------------
    // GrB_Scalar get/set
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_new (&s, GrB_FP32)) ;

    expected = GrB_NOT_IMPLEMENTED ;
    ERR (GrB_Scalar_get_VOID_(s, nothing, 0)) ;

    OK (GrB_Scalar_get_SIZE_(s, &size, GrB_ELTYPE_STRING)) ;
    CHECK (size == GxB_MAX_NAME_LEN) ;
    OK (GrB_Scalar_get_String_(s, name, GrB_ELTYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Scalar_get_String_(s, name, GrB_NAME)) ;
    CHECK (MATCH (name, "")) ;

    OK (GrB_Scalar_get_ENUM_(s, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    i = -1 ;
    OK (GrB_Scalar_get_Scalar_(s, s_int32, GrB_ELTYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GrB_FP32_CODE) ;

    GxB_print (s, 3) ;

    OK (GrB_Scalar_get_ENUM_(s, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    printf ("scalar storage: %d\n", i) ;
    CHECK (i == GrB_COLMAJOR) ;

    OK (GrB_Scalar_get_ENUM_(s, &i, GxB_FORMAT)) ;
    printf ("scalar storage: %d\n", i) ;
    CHECK (i == GxB_BY_COL) ;

    OK (GrB_Scalar_get_ENUM_(s, &i, GxB_SPARSITY_CONTROL)) ;
    printf ("sparsity control: %d\n", i) ;
    CHECK (i == GxB_AUTO_SPARSITY) ;

    GxB_print (s_int32, 3) ;
    OK (GrB_Scalar_get_ENUM_(s_int32, &i, GxB_SPARSITY_STATUS)) ;
    printf ("sparsity status: %d\n", i) ;
    CHECK (i == GxB_FULL) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Scalar_get_ENUM_(s_int32, &i, 0)) ;
    ERR (GrB_Scalar_get_SIZE_(s, &size, 0)) ;

    //--------------------------------------------------------------------------
    // GrB_Vector get/set
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&v, GrB_FP32, 10)) ;

    expected = GrB_NOT_IMPLEMENTED ;
    ERR (GrB_Vector_get_VOID_(v, nothing, 0)) ;

    OK (GrB_Vector_get_SIZE_(v, &size, GrB_ELTYPE_STRING)) ;
    CHECK (size == GxB_MAX_NAME_LEN) ;
    OK (GrB_Vector_get_String_(v, name, GrB_ELTYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Vector_get_String_(v, name, GrB_NAME)) ;
    CHECK (MATCH (name, "")) ;

    OK (GrB_Vector_get_ENUM_(v, &code, GrB_ELTYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    i = -1 ;
    OK (GrB_Vector_get_Scalar_(v, s_int32, GrB_ELTYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&i, s_int32)) ;
    CHECK (i == GrB_FP32_CODE) ;

    GxB_print (v, 3) ;

    OK (GrB_Vector_get_ENUM_(v, &i, GrB_STORAGE_ORIENTATION_HINT)) ;
    printf ("scalar storage: %d\n", i) ;
    CHECK (i == GrB_COLMAJOR) ;

    OK (GrB_Vector_get_ENUM_(v, &i, GxB_FORMAT)) ;
    printf ("scalar storage: %d\n", i) ;
    CHECK (i == GxB_BY_COL) ;

    OK (GrB_Vector_get_ENUM_(v, &i, GxB_SPARSITY_CONTROL)) ;
    printf ("sparsity control: %d\n", i) ;
    CHECK (i == GxB_AUTO_SPARSITY) ;

    OK (GrB_assign (v, NULL, NULL, 1, GrB_ALL, 10, NULL)) ;
    GxB_print (v, 3) ;

    OK (GrB_Vector_get_ENUM_(v, &i, GxB_SPARSITY_STATUS)) ;
    printf ("sparsity status: %d\n", i) ;
    CHECK (i == GxB_FULL) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Vector_get_ENUM_(v, &i, 0)) ;
    ERR (GrB_Vector_get_SIZE_(v, &size, 0)) ;

    fvalue = -1 ;
    OK (GrB_Vector_get_Scalar_(v, s_fp32, GxB_BITMAP_SWITCH)) ;
    OK (GrB_Scalar_extractElement_FP32_(&fvalue, s_fp32)) ;
    printf ("bitmap switch: %g\n", fvalue) ;
    CHECK (abs (fvalue - 0.04) < 1e-6) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GrB_free (&v) ;
    GrB_free (&s) ;
    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&unop) ;
    GrB_free (&binop) ;
    GrB_free (&idxunop) ;
    GrB_free (&desc) ;
    GrB_free (&monoid) ;
    GrB_free (&type) ;
    GrB_free (&semiring) ;
    GrB_free (&Context) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test26:  all tests passed\n\n") ;
}

