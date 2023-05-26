//------------------------------------------------------------------------------
// GrB_Monoid_set_*: set a field in a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Monoid_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_set_Scalar
(
    GrB_Monoid monoid,
    GrB_Scalar value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_set_String
( 
    GrB_Monoid monoid,
    char * value,
    GrB_Field field
)
{
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_set_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_set_ENUM
(
    GrB_Monoid monoid,
    int value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_set_VOID
(
    GrB_Monoid monoid,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

