//------------------------------------------------------------------------------
// GB_mex_reduce_to_GrB_Scalar: S = accum(S,reduce_to_scalar(A))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix or vector to a scalar

#include "GB_mex.h"

#define USAGE "S = GB_mex_reduce_to_GrB_Scalar (S, accum, reduce, A)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&S) ;              \
    GrB_Matrix_free_(&A) ;              \
    if (reduce_monoid_allocated)        \
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
    GrB_Matrix S = NULL ;
    GrB_Monoid reduce = NULL ;
    bool reduce_is_complex = false ;
    bool reduce_monoid_allocated = false ;

    // check inputs
    if (nargout > 1 || nargin != 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get the GrB_Scalar S as a GrB_Matrix
    #define GET_DEEP_COPY \
    S = GB_mx_mxArray_to_Matrix (pargin [0], "S input", true, true) ;
    #define FREE_DEEP_COPY GrB_Matrix_free_(&S) ;
    GET_DEEP_COPY ;
    if (S == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("S failed") ;
    }
    int64_t Snrows, Sncols ;
    GrB_Matrix_nrows (&Snrows, S) ;
    GrB_Matrix_ncols (&Sncols, S) ;
    if (Snrows != 1 || Sncols != 1)
    { 
        mexErrMsgTxt ("S must be a scalar") ;
    }
    GrB_Type stype = S->type ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [3], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get reduce
    bool user_complex = (Complex != GxB_FC64) && (stype == Complex) ;
    GrB_BinaryOp reduceop ;
    if (!GB_mx_mxArray_to_BinaryOp (&reduceop, pargin [2], "reduceop",
        stype, user_complex) || reduceop == NULL) 
    {
        FREE_ALL ;
        mexErrMsgTxt ("reduceop failed") ;
    }

    // get the reduce monoid
    if (user_complex)
    {
        if (reduceop == Complex_plus)
        {
            reduce = Complex_plus_monoid ;
        }
        else if (reduceop == Complex_times)
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
        if (!GB_mx_Monoid (&reduce, reduceop, malloc_debug))
        {
            FREE_ALL ;
            mexErrMsgTxt ("reduce failed") ;
        }
        reduce_monoid_allocated = true ;
    }

    // get accum, if present
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [1], "accum",
        stype, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // S = accum(S,A*B)

    // test both Vector and Matrix methods.  The typecast is not necessary,
    // just to test.
    if (A->vdim == 1)
    {
        GrB_Vector V = (GrB_Vector) A ;
        METHOD (GrB_Vector_reduce_Scalar ((GrB_Scalar) S, accum, reduce, V,
            NULL)) ;
    }
    else
    {
        METHOD (GrB_Matrix_reduce_Scalar ((GrB_Scalar) S, accum, reduce, A,
            NULL)) ;
    }

    // return S as struct
    pargout [0] = GB_mx_Matrix_to_mxArray (&S, "S", true) ;
    FREE_ALL ;
}

