//------------------------------------------------------------------------------
// gbmex_subassign: assign entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_subassign is an interface to GxB_Matrix_subassign and
// GxB_Matrix_subassign_[TYPE], for GrB.subassign and GhB.subassign.

// Usage for @GrB and @GhB (omitting desc argument):

// C = GrB.subassign (Cin, A, I, J)             C = Cin ; C(I,J) = A
// C = GrB.subassign (Cin, accum, A, I, J)      C = Cin ; C(I,J) += A
// C = GrB.subassign (Cin, M, A, I, J)          C = Cin ; C(I,J)<M> = A
// C = GrB.subassign (Cin, M, accum, A, I, J)   C = Cin ; C(I,J)<M> += A

// Usage for @GhB only (inplace):

// GhB.subassign (C, A, I, J)                   C(I,J) = A
// GhB.subassign (C, accum, A, I, J)            C(I,J) += A
// GhB.subassign (C, M, A, I, J)                C(I,J)<M> = A
// GhB.subassign (C, M, accum, A, I, J)         C(I,J)<M> += A

// A can be a matrix or a scalar.

#include "gb_interface.h"

#define USAGE "usage: C = GrB.subassign (Cin, M, accum, A, I, J, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    gbmx_assign_mexFunction (nargout, pargout, nargin, pargin, true, USAGE) ;
}

