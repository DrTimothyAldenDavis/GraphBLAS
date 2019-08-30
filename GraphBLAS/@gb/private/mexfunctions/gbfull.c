//------------------------------------------------------------------------------
// gbfull: convert a GraphBLAS matrix struct into a MATLAB dense matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard MATLAB
// sparse or dense matrix.  The output is a standard MATLAB dense matrix.

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

    gb_usage (nargin <= 4 && nargout <= 1,
        "usage: F = gbfull (X, type, id, desc)") ;

    //--------------------------------------------------------------------------
    // get a shallow copy of the input matrix
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, X)) ;
    OK (GrB_Matrix_ncols (&ncols, X)) ;

    //--------------------------------------------------------------------------
    // get the type of F
    //--------------------------------------------------------------------------

    GrB_Matrix type ;
    if (nargin > 1)
    {
        type = gb_mxstring_to_type (pargin [1]) ;
    }
    else
    {
        // the type of F defaults to the type of X
        OK (GxB_Matrix_type (&type, X)) ;
    }

    //--------------------------------------------------------------------------
    // get the identity scalar
    //--------------------------------------------------------------------------

    GrB_Matrix id ;
    if (nargin > 2)
    {
        id = gb_get_shallow (pargin [2]) ;
    }
    else
    {
        // assume the identity is zero, of the same type as F
        OK (GrB_Matrix_new (&id, type, 1, 1)) ;
    }

    //--------------------------------------------------------------------------
    // get the descriptor (kind defaults to KIND_FULL)
    //--------------------------------------------------------------------------

    kind_enum_t kind = KIND_FULL ;
    GxB_Format_Value fmt = GxB_NO_FORMAT ;
    GrB_Descriptor desc = NULL ;
    if (nargin > 3)
    {
        desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt) ;
    }

    // X determines the format of F, unless defined by the descriptor
    fmt = gb_get_format (nrows, ncols, X, NULL, fmt) ;

    //--------------------------------------------------------------------------
    // expand the identity into a dense matrix the same size as F
    //--------------------------------------------------------------------------

    GrB_Matrix Z ;
    OK (GrB_Matrix_new (&Z, type, nrows, ncols)) ;
    OK (GxB_set (Z, GxB_FORMAT, fmt)) ;
    gb_matrix_assign_scalar (Z, NULL, NULL, id, GrB_ALL, 0, GrB_ALL, 0, NULL,
        false) ;

    //--------------------------------------------------------------------------
    // F = first (X, Z)
    //--------------------------------------------------------------------------

    GrB_Matrix F ;
    OK (GrB_Matrix_new (&F, type, nrows, ncols)) ;
    OK (GxB_set (F, GxB_FORMAT, fmt)) ;
    OK (GrB_eWiseAdd (F, NULL, NULL, gb_first_binop (type), X, Z, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    OK (GrB_free (&id)) ;
    OK (GrB_free (&Z)) ;
    OK (GrB_free (&X)) ;
    OK (GrB_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export F to a MATLAB dense matrix
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&F, kind) ;
}

