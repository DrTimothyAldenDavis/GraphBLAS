//------------------------------------------------------------------------------
// gbmex_subassign: assign entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_subassign is an interface to GxB_Matrix_subassign and
// GxB_Matrix_assign_[TYPE], computing the GraphBLAS expression:

//      C(I,J)<#M,replace> = accum (C(I,J), A) or accum(C(I,J), A')

// where A can be a matrix or a scalar.

// Usage:

//      C = gbmex_subassign (ghb, Cin, M, accum, A, I, J, desc)

// Cin and A required.  See GrB.m for more details.
// C can be modified inplace.

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

