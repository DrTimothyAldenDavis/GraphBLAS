//------------------------------------------------------------------------------
// gbmex_emult: sparse matrix element-wise multiplication
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

#define USAGE "usage: C = GrB.emult (Cin, M, accum, binop, A, B, desc)"

// Usage for @GrB and @GhB (omitting desc argument):

// C = GrB.emult (op, A, B)                 C = op(A,B)
// C = GrB.emult (Cin, op, A, B)            C = op(A,B)
// C = GrB.emult (Cin, accum, op, A, B)     C = Cin + op(A,B)
// C = GrB.emult (Cin, M, op, A, B)         C = Cin ; C<M> = op(A,B)
// C = GrB.emult (Cin, M, accum, op, A, B)  C = Cin ; C<M> += op(A,B)

// Usage for @GhB only (inplace):

// GhB.emult (C, op, A, B)                  C = op(A,B)
// GhB.emult (C, accum, op, A, B)           C += op(A,B)
// GhB.emult (C, M, op, A, B)               C<M> = op(A,B)
// GhB.emult (C, M, accum, op, A, B)        C<M> += op(A,B)

// where op(A,B) refers to eWiseMult, A.*B, using the given op.

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

