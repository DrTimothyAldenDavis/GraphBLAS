//------------------------------------------------------------------------------
// GB_mex_extractTuples: extract all tuples from a matrix or vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[I,J,X] = GB_mex_extractTuples (A, xtype)"

#define FREE_ALL                        \
{                                       \
    GB_MATRIX_FREE (&A) ;               \
    GB_mx_put_global (true, 0) ;        \
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
    GB_void *Y = NULL ;
    GrB_Index nvals = 0 ;

    // check inputs
    GB_WHERE (USAGE) ;
    if (nargout > 3 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get the number of entries in A
    GrB_Matrix_nvals (&nvals, A) ;

    // create I
    pargout [0] = GB_mx_create_full (nvals, 1, GrB_UINT64) ;
    GrB_Index *I = (GrB_Index *) mxGetData (pargout [0]) ;

    // create J
    GrB_Index *J = NULL ;
    if (nargout > 1)
    {
        pargout [1] = GB_mx_create_full (nvals, 1, GrB_UINT64) ;
        J = (GrB_Index *) mxGetData (pargout [1]) ;
    }

    // create X
    GB_void *X = NULL ;
    GrB_Type xtype = GB_mx_string_to_Type (PARGIN (1), A->type) ;
    if (nargout > 2)
    {
        pargout [2] = GB_mx_create_full (nvals, 1, xtype) ;
        X = (GB_void *) mxGetData (pargout [2]) ;
    }

    // [I,J,X] = find (A)
    if (GB_VECTOR_OK (A))
    {
        // test extract vector methods
        GrB_Vector v = (GrB_Vector) A ;
        switch (xtype->code)
        {
            case GB_BOOL_code   : METHOD (GrB_Vector_extractTuples_BOOL   (I, (bool     *) X, &nvals, v)) ; break ;
            case GB_INT8_code   : METHOD (GrB_Vector_extractTuples_INT8   (I, (int8_t   *) X, &nvals, v)) ; break ;
            case GB_UINT8_code  : METHOD (GrB_Vector_extractTuples_UINT8  (I, (uint8_t  *) X, &nvals, v)) ; break ;
            case GB_INT16_code  : METHOD (GrB_Vector_extractTuples_INT16  (I, (int16_t  *) X, &nvals, v)) ; break ;
            case GB_UINT16_code : METHOD (GrB_Vector_extractTuples_UINT16 (I, (uint16_t *) X, &nvals, v)) ; break ;
            case GB_INT32_code  : METHOD (GrB_Vector_extractTuples_INT32  (I, (int32_t  *) X, &nvals, v)) ; break ;
            case GB_UINT32_code : METHOD (GrB_Vector_extractTuples_UINT32 (I, (uint32_t *) X, &nvals, v)) ; break ;
            case GB_INT64_code  : METHOD (GrB_Vector_extractTuples_INT64  (I, (int64_t  *) X, &nvals, v)) ; break ;
            case GB_UINT64_code : METHOD (GrB_Vector_extractTuples_UINT64 (I, (uint64_t *) X, &nvals, v)) ; break ;
            case GB_FP32_code   : METHOD (GrB_Vector_extractTuples_FP32   (I, (float    *) X, &nvals, v)) ; break ;
            case GB_FP64_code   : METHOD (GrB_Vector_extractTuples_FP64   (I, (double   *) X, &nvals, v)) ; break ;
            case GB_FC32_code   : METHOD (GxB_Vector_extractTuples_FC32   (I, (GxB_FC32_t *) X, &nvals, v)) ; break ;
            case GB_FC64_code   : METHOD (GxB_Vector_extractTuples_FC64   (I, (GxB_FC64_t *) X, &nvals, v)) ; break ;
            case GB_UDT_code    : METHOD (GrB_Vector_extractTuples_UDT    (I, (void     *) X, &nvals, v)) ; break ;
            default             : FREE_ALL ; mexErrMsgTxt ("unsupported type") ;
        }
        if (J != NULL)
        {
            for (int64_t p = 0 ; p < nvals ; p++) J [p] = 0 ;
        }
    }
    else
    {
        switch (xtype->code)
        {
            case GB_BOOL_code   : METHOD (GrB_Matrix_extractTuples_BOOL   (I, J, (bool     *) X, &nvals, A)) ; break ;
            case GB_INT8_code   : METHOD (GrB_Matrix_extractTuples_INT8   (I, J, (int8_t   *) X, &nvals, A)) ; break ;
            case GB_UINT8_code  : METHOD (GrB_Matrix_extractTuples_UINT8  (I, J, (uint8_t  *) X, &nvals, A)) ; break ;
            case GB_INT16_code  : METHOD (GrB_Matrix_extractTuples_INT16  (I, J, (int16_t  *) X, &nvals, A)) ; break ;
            case GB_UINT16_code : METHOD (GrB_Matrix_extractTuples_UINT16 (I, J, (uint16_t *) X, &nvals, A)) ; break ;
            case GB_INT32_code  : METHOD (GrB_Matrix_extractTuples_INT32  (I, J, (int32_t  *) X, &nvals, A)) ; break ;
            case GB_UINT32_code : METHOD (GrB_Matrix_extractTuples_UINT32 (I, J, (uint32_t *) X, &nvals, A)) ; break ;
            case GB_INT64_code  : METHOD (GrB_Matrix_extractTuples_INT64  (I, J, (int64_t  *) X, &nvals, A)) ; break ;
            case GB_UINT64_code : METHOD (GrB_Matrix_extractTuples_UINT64 (I, J, (uint64_t *) X, &nvals, A)) ; break ;
            case GB_FP32_code   : METHOD (GrB_Matrix_extractTuples_FP32   (I, J, (float    *) X, &nvals, A)) ; break ;
            case GB_FP64_code   : METHOD (GrB_Matrix_extractTuples_FP64   (I, J, (double   *) X, &nvals, A)) ; break;
            case GB_FC32_code   : METHOD (GxB_Matrix_extractTuples_FC32   (I, J, (GxB_FC32_t *) X, &nvals, A)) ; break ;
            case GB_FC64_code   : METHOD (GxB_Matrix_extractTuples_FC64   (I, J, (GxB_FC64_t *) X, &nvals, A)) ; break;
            case GB_UDT_code    : METHOD (GrB_Matrix_extractTuples_UDT    (I, J, (void     *) X, &nvals, A)) ; break;
            default             : FREE_ALL ; mexErrMsgTxt ("unsupported type") ;
        }
    }

    FREE_ALL ;
}

