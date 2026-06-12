//------------------------------------------------------------------------------
// gbmex_emult: sparse matrix element-wise multiplication
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_emult is an interface to GrB_Matrix_eWiseMult_BinaryOp.

// Usage:

// C = gbmex_emult (binop, A, B)
// C = gbmex_emult (binop, A, B, desc)
// C = gbmex_emult (Cin, accum, binop, A, B, desc)
// C = gbmex_emult (Cin, M, binop, A, B, desc)
// C = gbmex_emult (Cin, M, accum, binop, A, B, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A, B, and the descriptor).

#include "gb_interface.h"

#define USAGE "usage: C = GrB.emult (Cin, M, accum, binop, A, B, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    gbmx_ewise_mexFunction (nargout, pargout, nargin, pargin, false, USAGE) ;
}

