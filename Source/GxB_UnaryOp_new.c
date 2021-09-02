//------------------------------------------------------------------------------
// GxB_UnaryOp_new: create a new named unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// a unary operator: z = f (x).  The unary function signature must be
// void f (void *z, const void *x), and then it must recast its input and
// output arguments internally as needed.

#include "GB.h"

GrB_Info GxB_UnaryOp_new            // create a new user-defined unary operator
(
    GrB_UnaryOp *unaryop,           // handle for the new unary operator
    GxB_unary_function function,    // pointer to the unary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    const char *unop_name,          // name of the user function
    const char *unop_defn           // definition of the user function
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_UnaryOp_new (unaryop, function, ztype, xtype, name, defn)");
    GB_RETURN_IF_NULL (unaryop) ;
    (*unaryop) = NULL ;
    GB_RETURN_IF_NULL (function) ;
    GB_RETURN_IF_NULL_OR_FAULTY (ztype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (xtype) ;

    //--------------------------------------------------------------------------
    // create the unary op
    //--------------------------------------------------------------------------

    // allocate the unary operator
    size_t header_size ;
    (*unaryop) = GB_MALLOC (1, struct GB_UnaryOp_opaque, &header_size) ;
    if (*unaryop == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    // initialize the unary operator
    GrB_UnaryOp op = *unaryop ;
    op->magic = GB_MAGIC ;
    op->header_size = header_size ;
    op->xtype = xtype ;
    op->ztype = ztype ;
    op->function = function ;
    op->opcode = GB_USER_opcode ;   // user-defined operator
    // get the unary op name and defn
    GB_op_name_and_defn (op->name, &(op->defn), unop_name, unop_defn,
        "GxB_unary_function", 18) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_UNARYOP_OK (op, "new user-defined unary op", GB0) ;
    return (GrB_SUCCESS) ;
}
