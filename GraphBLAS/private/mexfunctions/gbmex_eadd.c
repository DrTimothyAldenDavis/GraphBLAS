//------------------------------------------------------------------------------
// gbmex_eadd: sparse matrix addition
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

#define USAGE "usage: C = GrB.eadd (Cin, M, accum, binop, A, B, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    gbmx_ewise_mexFunction (nargout, pargout, nargin, pargin, true, USAGE) ;
}

