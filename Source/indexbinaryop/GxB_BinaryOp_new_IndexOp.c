//------------------------------------------------------------------------------
// GxB_BinaryOp_new_IndexOp: create a new user-defined binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

// GxB_BinaryOp_new_IndexOp: create a new binary op from an index binary op
GrB_Info GxB_BinaryOp_new_IndexOp
(
    GrB_BinaryOp *binop_handle,     // handle of binary op to create
    GxB_IndexBinaryOp idxbinop,     // based on this index binary op
    GrB_Scalar theta                // theta value to bind to the new binary op
)
{
    int header_arena = GrB_DEFAULT ;
    return (GxB_BinaryOp_new_IndexOp_arena (binop_handle, idxbinop, theta,
        header_arena)) ;
}

