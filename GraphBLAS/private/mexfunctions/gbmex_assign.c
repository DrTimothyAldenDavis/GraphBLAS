//------------------------------------------------------------------------------
// gbmex_assign: assign entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_assign is an interface to GrB_Matrix_assign and
// GrB_Matrix_assign_[TYPE], computing the GraphBLAS expression:

//      C<#M,replace>(I,J) = accum (C(I,J), A) or accum(C(I,J), A')

// where A can be a matrix or a scalar.
// C can be modified inplace.

// Usage:

//      C = gbmex_assign (ghb, Cin, M, accum, A, I, J, desc)

// Cin and A required.  See GrB.m for more details.

#include "gb_interface.h"

#define USAGE "usage: C = GrB.assign (Cin, M, accum, A, I, J, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    gbmx_assign_mexFunction (nargout, pargout, nargin, pargin, false, USAGE) ;
}

