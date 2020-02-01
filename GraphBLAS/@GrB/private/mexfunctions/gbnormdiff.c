//------------------------------------------------------------------------------
// gbnormdiff: norm (A-B,kind)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

#define USAGE "usage: s = gbnormdiff (A, B, kind)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 3 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the inputs 
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Matrix B = gb_get_shallow (pargin [1]) ;
    int64_t norm_kind = gb_norm_kind (pargin [2]) ;

    GrB_Index anrows, ancols, bnrows, bncols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;
    if (anrows != bnrows || ancols != bncols)
    {
        ERROR ("A and B must have the same size") ;
    }

    //--------------------------------------------------------------------------
    // s = norm (A-B,kind)
    //--------------------------------------------------------------------------

    double s ;

    if (GB_is_dense (A) && GB_is_dense (B) &&
        (A->type == GrB_FP32 || A->type == GrB_FP64) && (A->type == B->type)
        && (anrows == 1 || ancols == 1))
    {
        // s = norm (A-B,p) where A and B are dense FP32 or FP64 vectors
        GrB_Index n ;
        s = GB_matlab_helper10 (A->x, B->x, A->type, norm_kind, anrows) ;
        if (s < 0) ERROR ("unknown norm") ;
    }
    else
    {
        ERROR ("TODO") ;
    }

    //--------------------------------------------------------------------------
    // s = norm (A-B,kind)
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&B)) ;
    pargout [0] = mxCreateDoubleScalar (s) ;
    GB_WRAPUP ;
}

