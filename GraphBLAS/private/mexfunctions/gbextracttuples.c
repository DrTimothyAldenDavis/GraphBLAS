//------------------------------------------------------------------------------
// gbextracttuples: extract all entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Usage:

// [I J X] = gb.extracttuples (A, desc)

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

    gb_usage (nargin >= 1 && nargin <= 2 && nargout <= 3,
        "usage: [I,J,X] = gb.extracttuples (A, desc)") ;

    //--------------------------------------------------------------------------
    // get the optional descriptor
    //--------------------------------------------------------------------------

    kind_enum_t kind = KIND_1BASED ;
    GrB_Descriptor desc = NULL ;
    if (nargin == 2)
    {
        desc = gb_mxarray_to_descriptor (pargin [1], &kind) ;
    }
    OK (GrB_free (&desc)) ;

    //--------------------------------------------------------------------------
    // get the matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    GrB_Type xtype ;
    OK (GxB_Matrix_type (&xtype, A)) ;

    //--------------------------------------------------------------------------
    // determine what to extract
    //--------------------------------------------------------------------------

    bool extract_I = true ;
    bool extract_J = (nargout > 1) ;
    bool extract_X = (nargout > 2) ;

    //--------------------------------------------------------------------------
    // allocate I and J
    //--------------------------------------------------------------------------

    GrB_Index s = MAX (nvals, 1) ;
    GrB_Index *I = extract_I ? mxMalloc (s * sizeof (GrB_Index)) : NULL ;
    GrB_Index *J = extract_J ? mxMalloc (s * sizeof (GrB_Index)) : NULL ;

    //--------------------------------------------------------------------------
    // extract the tuples
    //--------------------------------------------------------------------------

    if (xtype == GrB_BOOL)
    {
        bool *X = extract_X ? mxMalloc (s * sizeof (bool)) : NULL ;
        OK (GrB_Matrix_extractTuples_BOOL (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_BOOL) ;
        }
    }
    else if (xtype == GrB_INT8)
    {
        int8_t *X = extract_X ? mxMalloc (s * sizeof (int8_t)) : NULL ;
        OK (GrB_Matrix_extractTuples_INT8 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_INT8) ;
        }
    }
    else if (xtype == GrB_INT16)
    {
        int16_t *X = extract_X ? mxMalloc (s * sizeof (int16_t)) : NULL ;
        OK (GrB_Matrix_extractTuples_INT16 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_INT16) ;
        }
    }
    else if (xtype == GrB_INT32)
    {
        int32_t *X = extract_X ? mxMalloc (s * sizeof (int32_t)) : NULL ;
        OK (GrB_Matrix_extractTuples_INT32 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_INT32) ;
        }
    }
    else if (xtype == GrB_INT64)
    {
        int64_t *X = extract_X ? mxMalloc (s * sizeof (int64_t)) : NULL ;
        OK (GrB_Matrix_extractTuples_INT64 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_INT64) ;
        }
    }
    else if (xtype == GrB_UINT8)
    {
        uint8_t *X = extract_X ? mxMalloc (s * sizeof (uint8_t)) : NULL ;
        OK (GrB_Matrix_extractTuples_UINT8 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_UINT8) ;
        }
    }
    else if (xtype == GrB_UINT16)
    {
        uint16_t *X = extract_X ? mxMalloc (s * sizeof (uint16_t)) : NULL ;
        OK (GrB_Matrix_extractTuples_UINT16 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_UINT16) ;
        }
    }
    else if (xtype == GrB_UINT32)
    {
        uint32_t *X = extract_X ? mxMalloc (s * sizeof (uint32_t)) : NULL ;
        OK (GrB_Matrix_extractTuples_UINT32 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_UINT32) ;
        }
    }
    else if (xtype == GrB_UINT64)
    {
        uint64_t *X = extract_X ? mxMalloc (s * sizeof (uint64_t)) : NULL ;
        OK (GrB_Matrix_extractTuples_UINT64 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_UINT64) ;
        }
    }
    else if (xtype == GrB_FP32)
    {
        float *X = extract_X ? mxMalloc (s * sizeof (float)) : NULL ;
        OK (GrB_Matrix_extractTuples_FP32 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_FP32) ;
        }
    }
    else if (xtype == GrB_FP64)
    {
        double *X = extract_X ? mxMalloc (s * sizeof (double)) : NULL ;
        OK (GrB_Matrix_extractTuples_FP64 (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, GrB_FP64) ;
        }
    }
    #ifdef GB_COMPLEX_TYPE
    else if (xtype == gb_complex_type)
    {
        double *X = extract_X ? mxMalloc (s * sizeof (double complex)) : NULL ;
        OK (GrB_Matrix_extractTuples_UDT (I, J, X, &nvals, A)) ;
        if (extract_X)
        {
            pargout [2] = gb_export_to_mxfull (&X, nvals, 1, gb_complex_type) ;
        }
    }
    #endif
    else
    {
        ERROR ("unknown type") ;
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    OK (GrB_free (&A)) ;

    //--------------------------------------------------------------------------
    // export I and J
    //--------------------------------------------------------------------------

    if (kind == KIND_0BASED)
    {

        //----------------------------------------------------------------------
        // export I and J in their native zero-based integer format
        //----------------------------------------------------------------------

        if (extract_I)
        {
            pargout [0] = gb_export_to_mxfull (&I, nvals, 1, GrB_INT64) ;
        }
        if (extract_J)
        {
            pargout [1] = gb_export_to_mxfull (&J, nvals, 1, GrB_INT64) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // export I and J as double one-based integers (default)
        //----------------------------------------------------------------------

        // TODO do this in parallel
        if (extract_I)
        {
            double *I_double = mxMalloc (s * sizeof (double)) ;
            for (int64_t k = 0 ; k < nvals ; k++)
            {
                I_double [k] = (double) (I [k] + 1) ;
            }
            pargout [0] = gb_export_to_mxfull (&I_double, nvals, 1, GrB_FP64) ;
        }
        if (extract_J)
        {
            double *J_double = mxMalloc (s * sizeof (double)) ;
            for (int64_t k = 0 ; k < nvals ; k++)
            {
                J_double [k] = (double) (J [k] + 1) ;
            }
            pargout [1] = gb_export_to_mxfull (&J_double, nvals, 1, GrB_FP64) ;
        }
    }
}

