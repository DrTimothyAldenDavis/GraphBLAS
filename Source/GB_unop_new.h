//------------------------------------------------------------------------------
// GB_unop_new.h: create a new named unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_UNOP_NEW_H
#define GB_UNOP_NEW_H

GrB_Info GB_unop_new
(
    GrB_UnaryOp op,                 // new unary operator
    GxB_unary_function function,    // unary function (may be NULL)
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    const char *unop_name,          // name of the user function
    const char *unop_defn,          // definition of the user function
    const GB_Opcode opcode          // opcode for the function
) ;

#endif

