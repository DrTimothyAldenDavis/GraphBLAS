//------------------------------------------------------------------------------
// GB_mex_test37: index binary op tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

#define FREE_ALL                        \
{                                       \
    GrB_Scalar_free (&Theta) ;          \
    GrB_Scalar_free (&Alpha) ;          \
    GrB_Scalar_free (&Beta) ;           \
    GrB_Matrix_free (&A) ;              \
    GrB_Matrix_free (&A2) ;             \
    GrB_Matrix_free (&C1) ;             \
    GrB_Matrix_free (&C2) ;             \
    GrB_Matrix_free (&B1) ;             \
    GrB_Matrix_free (&B2) ;             \
    GrB_Matrix_free (&E1) ;             \
    GrB_Matrix_free (&E2) ;             \
    GrB_Matrix_free (&F1) ;             \
    GrB_Matrix_free (&F2) ;             \
    GrB_Matrix_free (&D) ;              \
    GrB_BinaryOp_free (&Bop) ;          \
    GzB_IndexBinaryOp_free (&Iop) ;     \
}

void test37_idxbinop (double *z,
    const double *x, GrB_Index ix, GrB_Index jx,
    const double *y, GrB_Index iy, GrB_Index jy,
    const double *theta) ;

void test37_idxbinop (double *z,
    const double *x, GrB_Index ix, GrB_Index jx,
    const double *y, GrB_Index iy, GrB_Index jy,
    const double *theta)
{
    (*z) = (*x) + 2*(*y) - 42*ix + jx + 3*iy + 1000*jy - (*theta) ;
}

#define TEST37_IDXBINOP                                                     \
"void test37_idxbinop (double *z,                                       \n" \
"    const double *x, GrB_Index ix, GrB_Index jx,                       \n" \
"    const double *y, GrB_Index iy, GrB_Index jy,                       \n" \
"    const double *theta)                                               \n" \
"{                                                                      \n" \
"    (*z) = (*x) + 2*(*y) - 42*ix + jx + 3*iy + 1000*jy - (*theta) ;    \n" \
"}                                                                      \n"

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
    // create index binary ops and test matrices
    //--------------------------------------------------------------------------

    GrB_Scalar Theta = NULL, Alpha = NULL, Beta = NULL ;
    GzB_IndexBinaryOp Iop = NULL ;
    GrB_BinaryOp Bop = NULL ;
    GrB_Matrix A = NULL, C1 = NULL, C2 = NULL, B1 = NULL, B2 = NULL, D = NULL,
        E1 = NULL, E2 = NULL, A2 = NULL, F1 = NULL, F2 = NULL ;

    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&C1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&C2, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&B1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&B2, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&E1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&E2, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&F1, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&F2, GrB_FP64, 10, 10)) ;

    // C1 and B1 always stay by column
    OK (GrB_Matrix_set_INT32 (C1, GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_set_INT32 (B1, GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;

    double x = 1 ;
    for (int64_t i = 0 ; i < 9 ; i++)
    {
        OK (GrB_Matrix_setElement_FP64 (A, x, i, i)) ;
        x = x*1.2 ;
        OK (GrB_Matrix_setElement_FP64 (A, x, i, i+1)) ;
        x = x*1.2 ;
        OK (GrB_Matrix_setElement_FP64 (A, x, i+1, i)) ;
        x = x*1.2 ;
    }
    OK (GrB_Matrix_setElement_FP64 (A, x, 9, 9)) ;
    x = x - 1000 ;
    OK (GrB_Matrix_setElement_FP64 (A, x, 5, 2)) ;

    OK (GrB_Scalar_new (&Theta, GrB_FP64)) ;
    OK (GrB_Scalar_setElement_FP64 (Theta, x)) ;

    // OK (GxB_print (A, 5)) ;
    // OK (GxB_print (Theta, 5)) ;

    OK (GzB_IndexBinaryOp_new2 (&Iop,
        (GzB_index_binary_function) test37_idxbinop,
        GrB_FP64, GrB_FP64, GrB_FP64, GrB_FP64,
        "test37_idxbinop", TEST37_IDXBINOP)) ;

    OK (GzB_IndexBinaryOp_set_String (Iop, "test37 idx binop", GrB_NAME)) ;
    OK (GxB_print (Iop, 5)) ;

    OK (GzB_BinaryOp_IndexOp_new (&Bop, Iop, Theta)) ;
    OK (GxB_print (Bop, 5)) ;

    OK (GrB_Scalar_new (&Alpha, GrB_FP64)) ;
    OK (GrB_Scalar_new (&Beta, GrB_FP64)) ;
    OK (GrB_Scalar_setElement_FP64 (Alpha, (double) 3.14159)) ;
    OK (GrB_Scalar_setElement_FP64 (Beta, (double) 42)) ;

    OK (GrB_Matrix_dup (&A2, A)) ;

    // OK (GrB_Global_set_INT32 (GrB_GLOBAL, 1 , (GrB_Field) GxB_BURBLE)) ;

    //--------------------------------------------------------------------------
    // test index binary ops
    //--------------------------------------------------------------------------

    for (int a1_sparsity = 0 ; a1_sparsity <= 1 ; a1_sparsity++)
    {
        for (int a2_sparsity = 0 ; a2_sparsity <= 1 ; a2_sparsity++)
        {
            for (int a1_store = 0 ; a1_store <= 1 ; a1_store++)
            {
                for (int a2_store = 0 ; a2_store <= 1 ; a2_store++)
                {
                    for (int c2_store = 0 ; c2_store <= 1 ; c2_store++)
                    {
                        for (int b2_store = 0 ; b2_store <= 1 ; b2_store++)
                        {
                            for (int jit = 0 ; jit <= 1 ; jit++)
                            {

                                printf (".") ;

                                // turn on/off the JIT
                                OK (GrB_Global_set_INT32 (GrB_GLOBAL,
                                    jit ? GxB_JIT_ON : GxB_JIT_OFF,
                                    (GrB_Field) GxB_JIT_C_CONTROL)) ;

                                // change A sparsity
                                OK (GrB_Matrix_set_INT32 (A,
                                    a1_sparsity ? GxB_SPARSE : GxB_BITMAP,
                                    (GrB_Field) GxB_SPARSITY_CONTROL)) ;

                                // change A storage orientation
                                OK (GrB_Matrix_set_INT32 (A,
                                    a1_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;

                                // C1 = add (A, A')
                                OK (GrB_Matrix_eWiseAdd_BinaryOp (C1,
                                    NULL, NULL, Bop, A, A, GrB_DESC_T1)) ;
                                // B1 = union (A, A')
                                OK (GxB_Matrix_eWiseUnion (B1, NULL, NULL, Bop,
                                    A, Alpha, A, Beta, GrB_DESC_T1)) ;
                                // E1 = emult (A, A')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (E1,
                                    NULL, NULL, Bop, A, A, GrB_DESC_T1)) ;
                                // F1 = emult (A, A')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (F1,
                                    NULL, NULL, Bop, A, A2, GrB_DESC_T1)) ;

                                // change A sparsity again
                                OK (GrB_Matrix_set_INT32 (A2,
                                    a2_sparsity ? GxB_SPARSE : GxB_BITMAP,
                                    (GrB_Field) GxB_SPARSITY_CONTROL)) ;

                                // change A storage again
                                OK (GrB_Matrix_set_INT32 (A,
                                    a2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;

                                // change C2, etc storage
                                OK (GrB_Matrix_set_INT32 (C2,
                                    c2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (B2,
                                    b2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (E2,
                                    b2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (F2,
                                    b2_store ? GrB_ROWMAJOR : GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;

                                // C2 = add (A, A')
                                OK (GrB_Matrix_eWiseAdd_BinaryOp (C2,
                                    NULL, NULL, Bop, A, A, GrB_DESC_T1)) ;
                                // B2 = union (A, A')
                                OK (GxB_Matrix_eWiseUnion (B2, NULL, NULL,
                                    Bop, A, Alpha, A, Beta, GrB_DESC_T1)) ;
                                // E2 = emult (A, A')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (E2,
                                    NULL, NULL, Bop, A, A, GrB_DESC_T1)) ;
                                // F2 = emult (A, A2')
                                OK (GrB_Matrix_eWiseMult_BinaryOp (F2,
                                    NULL, NULL, Bop, A, A2, GrB_DESC_T1)) ;

                                // OK (GxB_print (C1, 5)) ;
                                // OK (GxB_print (C2, 5)) ;

                                // OK (GxB_print (B1, 5)) ;
                                // OK (GxB_print (B2, 5)) ;

                                // change C2 etc to same storage as C1 etc
                                OK (GrB_Matrix_set_INT32 (C2, GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (B2, GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (E2, GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;
                                OK (GrB_Matrix_set_INT32 (F2, GrB_COLMAJOR,
                                    GrB_STORAGE_ORIENTATION_HINT)) ;

                                // FIXME: check C1, etc matrices

                                OK (GrB_Matrix_new (&D, GrB_FP64, 10, 10)) ;
                                OK (GrB_Matrix_eWiseAdd_BinaryOp (D, NULL,
                                    NULL, GrB_MINUS_FP64, C1, B1, NULL)) ;
                                OK (GrB_Matrix_select_FP64 (D, NULL, NULL,
                                    GrB_VALUENE_FP64, D, (double) 0, NULL)) ;
                                // OK (GxB_print (D, 5)) ;
                                OK (GrB_Matrix_free (&D)) ;

                                CHECK (GB_mx_isequal (C1, C2, 0)) ;
                                CHECK (GB_mx_isequal (B1, B2, 0)) ;
                                CHECK (GB_mx_isequal (E1, E2, 0)) ;
                                CHECK (GB_mx_isequal (F1, F2, 0)) ;
                                CHECK (GB_mx_isequal (F1, E2, 0)) ;
                            }
                        }
                    }
                }
            }
        }
    }

    //------------------------------------------------------------------------
    // error tests
    //------------------------------------------------------------------------

    printf ("\nerror handling tests:\n") ;

    int expected = GrB_INVALID_OBJECT ;
    void *p = Bop->theta_type = NULL ;
    Bop->theta_type = NULL ;
    ERR (GB_BinaryOp_check (Bop, "Bop: bad theta_type", 5, stdout)) ;
    Bop->theta_type = p ;

    p = Iop->idxbinop_function ;
    Iop->idxbinop_function = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null function", 5, stdout)) ;
    Iop->idxbinop_function = p ;

    p = Iop->ztype ;
    Iop->ztype = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null ztype", 5, stdout)) ;
    Iop->ztype = p ;

    p = Iop->xtype ;
    Iop->xtype = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null xtype", 5, stdout)) ;
    Iop->xtype = p ;

    p = Iop->ytype ;
    Iop->ytype = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null ytype", 5, stdout)) ;
    Iop->ytype = p ;

    p = Iop->theta_type ;
    Iop->theta_type = NULL ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: null theta_type", 5, stdout)) ;
    Iop->theta_type = p ;

    GB_Opcode code = Iop->opcode ;
    Iop->opcode = GB_PLUS_binop_code ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: invalid opcode", 5, stdout)) ;
    Iop->opcode = code ;

    int len = Iop->name_len ;
    Iop->name_len = 3 ;
    ERR (GB_IndexBinaryOp_check (Iop, "Iop: invalid name_len", 5, stdout)) ;
    Iop->name_len = len ;

    expected = GrB_NULL_POINTER ;
    ERR (GB_IndexBinaryOp_check (NULL, "Iop: null", 5, stdout)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GzB_IndexBinaryOp_set_Scalar (Iop, Theta, GrB_NAME)) ;
    ERR (GzB_IndexBinaryOp_set_INT32 (Iop, 2, GrB_SIZE)) ;
    ERR (GzB_IndexBinaryOp_set_VOID (Iop, NULL, GrB_SIZE, 0)) ;

    //------------------------------------------------------------------------
    // finalize GraphBLAS
    //------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test37:  all tests passed\n\n") ;
}

