//------------------------------------------------------------------------------
// gbbuild: build a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Usage:

// A = gbbuild (I, J, X)
// A = gbbuild (I, J, X, m)
// A = gbbuild (I, J, X, m, n)
// A = gbbuild (I, J, X, m, n, dup,
// A = gbbuild (I, J, X, m, n, dup, type) ;

// m and n default to the largest index in I and J, respectively.
// dup defaults to 'plus.xtype' where xtype is the type of X.
// If dup is given by without a type,  type of dup defaults to the type of X.
// type is the type of A, which defaults to the type of X.

#include "gb_matlab.h"

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

    gb_usage (nargin >= 3 && nargin <= 7 && nargin != 4 && nargout <= 1,
        "usage: A = gbbuild (I, J, X, m, n, dup, type)") ;

    //--------------------------------------------------------------------------
    // get I and J
    //--------------------------------------------------------------------------

    GrB_Index ni, Icolon [3], *I = NULL, I_is_list, I_is_allocated ;
    GrB_Index nj, Jcolon [3], *J = NULL, J_is_list, J_is_allocated ;

    // TODO: if I,J allocated, also find max index
    int64_t I_max, J_max ;

    gb_mxarray_to_indices (&I, pargin [0], &ni, Icolon, &I_is_list,
        &I_is_allocated, &I_max) ;

    gb_mxarray_to_indices (&J, pargin [1], &nj, Jcolon, &J_is_list,
        &J_is_allocated, &J_max) ;

    CHECK_ERROR (!I_is_list, "I must be a list of indices") ;
    CHECK_ERROR (!J_is_list, "J must be a list of indices") ;
    CHECK_ERROR (ni != nj, "I, J, and X must be the same size") ;

    //--------------------------------------------------------------------------
    // get X
    //--------------------------------------------------------------------------

    GrB_Type xtype = gb_mxarray_type (pargin [2]) ;

    GrB_Index nvals = mxGetNumberOfElements (pargin [2]) ;

    CHECK_ERROR (ni != nvals, "I, J, and X must be the same size") ;
    CHECK_ERROR (!(mxIsNumeric (pargin [2]) || mxIsLogical (pargin [2])),
        "X must be a numeric or logical array") ;
    CHECK_ERROR (mxIsSparse (pargin [2]), "X cannot be sparse") ;

    // void *X = mxGetData (pargin [2]) ;

    //--------------------------------------------------------------------------
    // get m and n if present
    //--------------------------------------------------------------------------

    GrB_Index nrows = 0, ncols = 0 ;

    if (nargin < 4)
    {
        // nrows = max entry in I + 1
        for (int64_t k = 0 ; k < (int64_t) ni ; k++)
        {
            nrows = MAX (nrows, I [k]) ;
        }
        if (ni > 0) nrows++ ;
    }
    else
    {
        // m is provided on input
        CHECK_ERROR (!IS_SCALAR (pargin [3]), "m must be a scalar") ;
        nrows = (GrB_Index) mxGetScalar (pargin [3]) ;
    }

    if (nargin < 5)
    {
        // ncols = max entry in J
        for (int64_t k = 0 ; k < (int64_t) ni ; k++)
        {
            ncols = MAX (ncols, I [k]) ;
        }
        if (ni > 0) ncols++ ;
    }
    else
    {
        // n is provided on input
        CHECK_ERROR (!IS_SCALAR (pargin [4]), "n must be a scalar") ;
        ncols = (GrB_Index) mxGetScalar (pargin [4]) ;
    }

    //--------------------------------------------------------------------------
    // get the dup operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp dup = NULL ;
    if (nargin > 5)
    {
        dup = gb_mxstring_to_binop (pargin [5], xtype) ;
    }

    // if dup is NULL, defaults to plus.xtype, below.

    //--------------------------------------------------------------------------
    // get the output matrix type
    //--------------------------------------------------------------------------

    GrB_Type type = NULL ;
    if (nargin > 6)
    {
        type = gb_mxstring_to_type (pargin [6]) ;
    }
    if (type == NULL) type = xtype ;

    //--------------------------------------------------------------------------
    // build the matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A ;
    OK (GrB_Matrix_new (&A, type, nrows, ncols)) ;

    if (xtype == GrB_BOOL)
    {
        bool *X = mxGetData (pargin [2]) ;
        if (dup == NULL) dup = GrB_LOR ;
        OK (GrB_Matrix_build_BOOL (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_INT8)
    {
        int8_t *X = mxGetInt8s (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_INT8 ;
        OK (GrB_Matrix_build_INT8 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_INT16)
    {
        int16_t *X = mxGetInt16s (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_INT16 ;
        OK (GrB_Matrix_build_INT16 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_INT32)
    {
        int32_t *X = mxGetInt32s (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_INT32 ;
        OK (GrB_Matrix_build_INT32 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_INT64)
    {
        int64_t *X = mxGetInt64s (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_INT64 ;
        OK (GrB_Matrix_build_INT64 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_UINT8)
    {
        uint8_t *X = mxGetUint8s (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_UINT8 ;
        OK (GrB_Matrix_build_UINT8 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_UINT16)
    {
        uint16_t *X = mxGetUint16s (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_UINT16 ;
        OK (GrB_Matrix_build_UINT16 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_UINT32)
    {
        uint32_t *X = mxGetUint32s (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_UINT32 ;
        OK (GrB_Matrix_build_UINT32 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_UINT64)
    {
        uint64_t *X = mxGetUint64s (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_UINT64 ;
        OK (GrB_Matrix_build_UINT64 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_FP32)
    {
        float *X = mxGetSingles (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_FP32 ;
        OK (GrB_Matrix_build_FP32 (A, I, J, X, nvals, dup)) ;
    }
    else if (xtype == GrB_FP64)
    {
        double *X = mxGetDoubles (pargin [2]) ;
        if (dup == NULL) dup = GrB_PLUS_FP64 ;
        OK (GrB_Matrix_build_FP64 (A, I, J, X, nvals, dup)) ;
    }
    #ifdef GB_COMPLEX_TYPE
    else if (xtype == gb_complex_type)
    {
        double *X = mxGetComplexDoubles (pargin [2]) ;
        if (dup == NULL) dup = ... ;
        OK (GrB_Matrix_build_UDT (A, I, J, X, nvals, dup)) ;
    }
    #endif
    else
    {
        ERROR ("unknown type") ;
    }

    //--------------------------------------------------------------------------
    // export the output matrix A back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_matrix_to_mxstruct (&A) ;
}

