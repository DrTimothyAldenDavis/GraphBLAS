//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_*: set a field in a unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_set_Scalar
(
    GrB_IndexUnaryOp op,
    GrB_Scalar value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_set_String
(
    GrB_IndexUnaryOp op,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_IndexUnaryOp_set_String (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for get", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_op_string_set ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_set_ENUM
(
    GrB_IndexUnaryOp op,
    int value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_IndexUnaryOp_set_VOID
(
    GrB_IndexUnaryOp op,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

