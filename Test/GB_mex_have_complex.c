//------------------------------------------------------------------------------
// GB_mex_have_complex: determine if the 'double complex' type is available
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mex.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    #if HAVE_COMPLEX
        pargout [0] = mxCreateLogicalScalar (true) ;
    #else
        pargout [0] = mxCreateLogicalScalar (false) ;
    #endif
}

