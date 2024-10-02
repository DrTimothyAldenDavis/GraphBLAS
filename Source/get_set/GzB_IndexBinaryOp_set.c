//------------------------------------------------------------------------------
// GzB_IndexBinaryOp_set_*: set a field in a index binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GzB_IndexBinaryOp_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GzB_IndexBinaryOp_set_Scalar
(
    GzB_IndexBinaryOp op,
    GrB_Scalar value,
    GrB_Field field
)
{ 
GB_GOTCHA ;
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GzB_IndexBinaryOp_set_String
//------------------------------------------------------------------------------

GrB_Info GzB_IndexBinaryOp_set_String
(
    GzB_IndexBinaryOp op,
    char * value,
    GrB_Field field
)
{ 
GB_GOTCHA ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GzB_IndexBinaryOp_set_String (op, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_INDEXBINARYOP_OK (op, "idxbinop for set", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_op_string_set ((GB_Operator) op, value, field)) ;
}

//------------------------------------------------------------------------------
// GzB_IndexBinaryOp_set_INT32
//------------------------------------------------------------------------------

GrB_Info GzB_IndexBinaryOp_set_INT32
(
    GzB_IndexBinaryOp op,
    int32_t value,
    GrB_Field field
)
{ 
GB_GOTCHA ;
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GzB_IndexBinaryOp_set_VOID
//------------------------------------------------------------------------------

GrB_Info GzB_IndexBinaryOp_set_VOID
(
    GzB_IndexBinaryOp op,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
GB_GOTCHA ;
    return (GrB_INVALID_VALUE) ;
}

