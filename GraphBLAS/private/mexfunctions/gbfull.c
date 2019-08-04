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

    gb_usage (nargin <= 2 && nargout <= 1,
        "usage: F = full (X) or full (X, identity") ;

    //--------------------------------------------------------------------------
    // get a shallow copy of the input matrix
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;
    GrB_Index nrows, ncols ;
    GrB_Type xtype ;
    OK (GrB_Matrix_nrows (&nrows, X)) ;
    OK (GrB_Matrix_ncols (&ncols, X)) ;
    OK (GxB_Matrix_type (&xtype, X)) ;

    //--------------------------------------------------------------------------
    // get the identity scalar
    //--------------------------------------------------------------------------

    GrB_Matrix id ;
    if (nargin > 1)
    {
        CHECK_ERROR (!mxIsScalar (pargin [1]), "identity must be a scalar") ;
        CHECK_ERROR (mxIsSparse (pargin [1]), "identity must not be sparse") ;
        id = gb_get_shallow (pargin [1]) ;
    }
    else
    {
        // assume the identity is zero, of the same type as X
        OK (GrB_Matrix_new (&id, xtype, 1, 1)) ;
        OK (GrB_Matrix_setElement (id, 0, 0, 0)) ;
    }

    //--------------------------------------------------------------------------
    // expand the identity into a matrix the same size as X
    //--------------------------------------------------------------------------

    GrB_Matrix Z ;
    GrB_Type idtype ;
    OK (GxB_Matrix_type (&idtype, id)) ;
    OK (GrB_Matrix_new (&Z, idtype, nrows, ncols)) ;
    gb_matrix_assign_scalar (Z, NULL, NULL, id, GrB_ALL, 0, GrB_ALL, 0, NULL) ;

    //--------------------------------------------------------------------------
    // C = X + Z using the FIRST operator
    //--------------------------------------------------------------------------

    GrB_Matrix C ;
    OK (GrB_Matrix_new (&C, xtype, nrows, ncols)) ;
    OK (GrB_eWiseAdd (C, NULL, NULL, gb_first_binop (xtype), X, Z, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    OK (GrB_free (&id)) ;
    OK (GrB_free (&Z)) ;
    OK (GrB_free (&X)) ;

    //--------------------------------------------------------------------------
    // export C to a MATLAB dense matrix
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_FULL) ;

}

