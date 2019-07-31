//------------------------------------------------------------------------------
// gbsparse: convert a GraphBLAS matrix struct into a MATLAB sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard MATLAB
// sparse matrix.  The output is a standard MATLAB sparse matrix.

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

    gb_usage (nargin == 1 && nargout <= 1, "usage: A = gbsparse (X)") ;

    //--------------------------------------------------------------------------
    // convert the input matrix to a MATLAB sparse matrix
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;
    pargout [0] = gb_export_to_mxarray (&X) ;
}

