//------------------------------------------------------------------------------
// gbmex_eadd: sparse matrix addition
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_eadd is an interface to GrB_Matrix_eWiseAdd_BinaryOp.

// Usage:

// C = gbmex_eadd (ghb, binop, A, B)
// C = gbmex_eadd (ghb, binop, A, B, desc)
// C = gbmex_eadd (ghb, Cin, accum, binop, A, B, desc)
// C = gbmex_eadd (ghb, Cin, M, binop, A, B, desc)
// C = gbmex_eadd (ghb, Cin, M, accum, binop, A, B, desc)

// TODO: add in-place syntax
// gbmex_eadd (ghb, C, accum, binop, A, B, desc)
// gbmex_eadd (ghb, C, M, binop, A, B, desc)
// gbmex_eadd (ghb, C, M, accum, binop, A, B, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A, B, and the descriptor).

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

