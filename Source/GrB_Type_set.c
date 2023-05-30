//------------------------------------------------------------------------------
// GrB_Type_set_*: set a field in a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Type_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_Scalar
(
    GrB_Type type,
    GrB_Scalar value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Type_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_String
(
    GrB_Type type,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Type_set_String (type, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_TYPE_OK (type, "unaryop for get", GB0) ;

    //--------------------------------------------------------------------------
    // set the name or defn of a user-defined type
    //--------------------------------------------------------------------------

    return (GB_op_or_type_string_set (type->code == GB_UDT_code, true, value,
        field, type->name, &(type->name_len), &(type->defn),
        &(type->defn_size), &(type->hash))) ;
}

//------------------------------------------------------------------------------
// GrB_Type_set_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_ENUM
(
    GrB_Type type,
    int value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Type_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_VOID
(
    GrB_Type type,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

