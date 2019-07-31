//------------------------------------------------------------------------------
// gbsize: number of rows and columns in a GraphBLAS matrix struct
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

    gb_usage (nargin == 1 && nargout <= 2, "usage: [m n] = gbsize (X)") ;

    //--------------------------------------------------------------------------
    // get the # of rows and columns in a GraphBLAS matrix struct
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;

    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, X)) ;
    OK (GrB_Matrix_ncols (&ncols, X)) ;

    if (nargout <= 1)
    {
        pargout [0] = mxCreateDoubleMatrix (1, 2, mxREAL) ;
        double *p = mxGetDoubles (pargout [0]) ;
        p [0] = (double) nrows ;
        p [1] = (double) ncols ;
    }
    else
    {
        pargout [0] = mxCreateDoubleScalar ((double) nrows) ;
        pargout [1] = mxCreateDoubleScalar ((double) ncols) ;
    }

    OK (GrB_free (&X)) ;
}

