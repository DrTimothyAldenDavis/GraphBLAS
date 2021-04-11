//------------------------------------------------------------------------------
// GB_mex_reduce_to_vector: c = accum(c,reduce_to_vector(A))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix to a vector: w<mask> = accum (w, reduce_to_vector (A))

// MATLAB interface to GrB_reduce, which relies on
// GrB_Matrix_reduce_Monoid to reduce a matrix to a vector.

#include "GB_mex.h"

#define USAGE "w = GB_mex_reduce_to_vector (w, mask, accum, reduce, A, desc)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Vector_free_(&w) ;              \
    GrB_Vector_free_(&mask) ;           \
    GrB_Descriptor_free_(&desc) ;       \
    if (!user_complex)                  \
    {                                   \
        GrB_Monoid_free_(&reduce) ;     \
    }                                   \
    GB_mx_put_global (true) ;           \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Vector w = NULL ;
    GrB_Vector mask = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Monoid reduce = NULL ;
    bool user_complex = false ;

    // check inputs
    if (nargout > 1 || nargin < 5 || nargin > 6)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get w (make a deep copy)
    #define GET_DEEP_COPY \
    w = GB_mx_mxArray_to_Vector (pargin [0], "w input", true, true) ;
    #define FREE_DEEP_COPY GrB_Vector_free_(&w) ;
    GET_DEEP_COPY ;
    if (w == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("w failed") ;
    }

    // get mask (shallow copy)
    mask = GB_mx_mxArray_to_Vector (pargin [1], "mask", false, false) ;
    if (mask == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("mask failed") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [4], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get reduce operator
    user_complex = (Complex != GxB_FC64) && (A->type == Complex) ;
    GrB_BinaryOp op ;
    if (!GB_mx_mxArray_to_BinaryOp (&op, pargin [3], "op",
        w->type, user_complex) || op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("op failed") ;
    }

    // get the reduce monoid
    if (user_complex)
    {
        if (op == Complex_plus)
        {
            reduce = Complex_plus_monoid ;
        }
        else if (op == Complex_times)
        {
            reduce = Complex_times_monoid ;
        }
        else
        {
            FREE_ALL ;
            mexErrMsgTxt ("reduce failed") ;
        }
    }
    else
    {
        // create the reduce monoid
        if (!GB_mx_Monoid (&reduce, op, malloc_debug))
        {
            FREE_ALL ;
            mexErrMsgTxt ("reduce failed") ;
        }
    }

    // get accum, if present
    user_complex = (Complex != GxB_FC64) && (w->type == Complex) ;
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        w->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (5), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // w<mask> = accum (w, reduce_to_vector (A)) using a monoid
    METHOD (GrB_Matrix_reduce_Monoid_(w, mask, accum, reduce, A, desc)) ;

    // return w to MATLAB as a struct and free the GraphBLAS w
    pargout [0] = GB_mx_Vector_to_mxArray (&w, "w output", true) ;

    FREE_ALL ;
}

