//------------------------------------------------------------------------------
// GB_BinaryOp_new: create a new user-defined binary operator (historical)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_BinaryOp_new
(
    GrB_BinaryOp *binaryop,         // handle for the new binary operator
    GxB_binary_function function,   // pointer to the binary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    GrB_Type ytype,                 // type of input y
    const char *name                // name of the user function
)
{ 
    return (GxB_BinaryOp_new (binaryop, function, ztype, xtype, ytype,
        name, NULL)) ;
}

