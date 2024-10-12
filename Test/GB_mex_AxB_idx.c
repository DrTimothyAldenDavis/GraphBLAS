//------------------------------------------------------------------------------
// GB_mex_AxB_idx: C=A*B, A'*B, A*B', or A'*B' using the (MIN,SECONDI1) semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is for testing only.  See GrB_mxm instead.

// FIXME: add other monoids
// FIXME: add other multops: firsti, etc

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "C = GB_mex_AxB_idx (A, B, atrans, btrans, axb_method," \
    " C_is_csc, builtin)"

#define FREE_ALL                                \
{                                               \
    GrB_Matrix_free (&A) ;                      \
    GrB_Matrix_free (&B) ;                      \
    GrB_Matrix_free (&C) ;                      \
    GrB_Scalar_free (&Theta) ;                  \
    GrB_Descriptor_free (&desc) ;               \
    GrB_BinaryOp_free (&mult) ;                 \
    GzB_IndexBinaryOp_free (&Secondi1) ;        \
    GrB_Semiring_free (&semiring) ;             \
    GB_mx_put_global (true) ;                   \
}

//------------------------------------------------------------------------------
// user-defined SECONDI1 index binary operator
//------------------------------------------------------------------------------

void secondi1_idxbinop (int64_t *z,
    const void *x, GrB_Index ix, GrB_Index jx,
    const void *y, GrB_Index iy, GrB_Index jy,
    const void *theta) ;

void secondi1_idxbinop (int64_t *z,
    const void *x, GrB_Index ix, GrB_Index jx,
    const void *y, GrB_Index iy, GrB_Index jy,
    const void *theta)
{
    (*z) = iy + 1 ;
}

#define SECONDI1_IDXBINOP                               \
"void secondi1_idxbinop (int64_t *z,                \n" \
"    const void *x, GrB_Index ix, GrB_Index jx,     \n" \
"    const void *y, GrB_Index iy, GrB_Index jy,     \n" \
"    const void *theta)                             \n" \
"{                                                  \n" \
"    (*z) = iy + 1 ;                                \n" \
"}"

//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info = GrB_SUCCESS ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GrB_Scalar Theta = NULL ;
    GrB_BinaryOp mult = NULL ;
    GzB_IndexBinaryOp Secondi1 = NULL ;
    GrB_Semiring semiring = NULL ;
    GrB_Index anrows = 0, ancols = 0, bnrows = 0, bncols = 0 ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 7)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A and B
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
    B = GB_mx_mxArray_to_Matrix (pargin [1], "B", false, true) ;
    if (A == NULL || B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("failed") ;
    }

    // get the atrans option
    bool GET_SCALAR (2, bool, atrans, false) ;

    // get the btrans option
    bool GET_SCALAR (3, bool, btrans, false) ;

    // get the axb_method
    GrB_Desc_Value GET_SCALAR (4, GrB_Desc_Value, AxB_method, GxB_DEFAULT) ;

    // get the C_is_csc option
    bool GET_SCALAR (5, bool, C_is_csc, true) ;

    // get the builtin option
    bool GET_SCALAR (6, bool, builtin, true) ;

    // set the Descriptor
    OK (GrB_Descriptor_new (&desc)) ;
    OK (GrB_Descriptor_set (desc, GrB_INP0, atrans ? GrB_TRAN : GxB_DEFAULT)) ;
    OK (GrB_Descriptor_set (desc, GrB_INP1, btrans ? GrB_TRAN : GxB_DEFAULT)) ;
    OK (GrB_Descriptor_set (desc, GxB_AxB_METHOD, AxB_method)) ;

    // determine the dimensions
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;
    GrB_Index cnrows = (atrans) ? ancols : anrows ;
    GrB_Index cncols = (btrans) ? bnrows : bncols ;

    // create the output matrix C
    OK (GrB_Matrix_new (&C, GrB_INT64, cnrows, cncols)) ;
    OK (GrB_Matrix_set_INT32 (C, C_is_csc, GrB_STORAGE_ORIENTATION_HINT)) ;

    // create the semiring
    if (builtin)
    {
        // builtin (MIN,SECONDI1)
        semiring = GxB_MIN_SECONDI1_INT64 ;
    }
    else
    {
        // user-defined (MIN,SECONDI1)
        OK (GzB_IndexBinaryOp_new2 (&Secondi1,
            (GzB_index_binary_function) secondi1_idxbinop,
            GrB_INT64, GrB_FP64, GrB_FP64, GrB_FP64,
            "secondi1_idxbinop", SECONDI1_IDXBINOP)) ;
        OK (GrB_Scalar_new (&Theta, GrB_INT64)) ;
        OK (GrB_Scalar_setElement_INT64 (Theta, 1)) ;
        OK (GzB_BinaryOp_IndexOp_new (&mult, Secondi1, Theta)) ;
        OK (GrB_Semiring_new (&semiring, GrB_MIN_MONOID_INT64, mult)) ;
    }

    // C = A*B, A'*B, A*B', or A'*B'
    OK (GrB_mxm (C, NULL, NULL, semiring, A, B, desc)) ;

    // return C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C AxB idx result", true) ;
    FREE_ALL ;
}

