//------------------------------------------------------------------------------
// GB_mex_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2018, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "GB_mex_init"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    GB_WHERE (USAGE) ;
    GxB_init (GrB_NONBLOCKING, mxMalloc, mxCalloc, mxRealloc, mxFree) ;
    GB_Global.abort_function = GB_mx_abort ;
    GxB_set (GxB_FORMAT, GxB_BY_COL) ;
}

