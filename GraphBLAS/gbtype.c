//------------------------------------------------------------------------------
// gbtype: type of a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard MATLAB
// sparse matrix.

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

    gb_usage (nargin == 1 && nargout <= 1, "usage: type = gbtype (X)") ;

    //--------------------------------------------------------------------------
    // get the # of entries in the matrix
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, X)) ;

         if (type == GrB_BOOL  ) pargout [0] = mxCreateString ("logical") ;
    else if (type == GrB_INT8  ) pargout [0] = mxCreateString ("int8") ;
    else if (type == GrB_INT16 ) pargout [0] = mxCreateString ("int16") ;
    else if (type == GrB_INT32 ) pargout [0] = mxCreateString ("int32") ;
    else if (type == GrB_INT64 ) pargout [0] = mxCreateString ("int64") ;
    else if (type == GrB_UINT8 ) pargout [0] = mxCreateString ("uint8") ;
    else if (type == GrB_UINT16) pargout [0] = mxCreateString ("uint16") ;
    else if (type == GrB_UINT32) pargout [0] = mxCreateString ("uint32") ;
    else if (type == GrB_UINT64) pargout [0] = mxCreateString ("uint64") ;
    else if (type == GrB_FP32  ) pargout [0] = mxCreateString ("single") ;
    else if (type == GrB_FP64  ) pargout [0] = mxCreateString ("double") ;
    #ifdef GB_COMPLEX_TYPE
    else if (type == gb_complex_type) pargout [0] = mxCreateString ("complex") ;
    #endif
    else ERROR ("unknown type") ;

    gb_free_shallow (&X) ;
}

