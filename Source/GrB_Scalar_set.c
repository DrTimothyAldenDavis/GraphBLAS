//------------------------------------------------------------------------------
// GrB_Scalar_set_*: set a field in a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Scalar_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_Scalar
(
    GrB_Scalar v,
    GrB_Scalar value,
    GrB_Field field
)
{
    // all settings are ignored
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_String
(
    GrB_Scalar v,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Scalar_set_String (v, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_VECTOR_OK (v, "v to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GrB_NOT_IMPLEMENTED) ;      // TODO: set the name
}

//------------------------------------------------------------------------------
// GrB_Scalar_set_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_ENUM
(
    GrB_Scalar v,
    int value,
    GrB_Field field
)
{
    // all settings are ignored
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_VOID
(
    GrB_Scalar v,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

