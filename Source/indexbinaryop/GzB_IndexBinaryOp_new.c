//------------------------------------------------------------------------------
// GzB_IndexBinaryOp_new: create a new user-defined index_binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GzB_IndexBinaryOp_new
(
    GzB_IndexBinaryOp *op,          // handle for the new index binary operator
    GzB_index_binary_function function, // pointer to the index binary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    GrB_Type ytype,                 // type of input y
    GrB_Type theta_type             // type of input theta
)
{ 
// GB_GOTCHA ;  new index binary op
    return (GzB_IndexBinaryOp_new2 (op, function, ztype, xtype, ytype,
        theta_type, NULL, NULL)) ;
}

