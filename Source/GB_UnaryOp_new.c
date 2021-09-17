//------------------------------------------------------------------------------
// GB_UnaryOp_new: create a new user-defined unary operator (historical)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_UnaryOp_new             // create a new user-defined unary operator
(
    GrB_UnaryOp *unaryop,           // handle for the new unary operator
    GxB_unary_function function,    // pointer to the unary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    const char *name                // name of the user function
)
{ 
    return (GxB_UnaryOp_new (unaryop, function, ztype, xtype, name, NULL)) ;
}

