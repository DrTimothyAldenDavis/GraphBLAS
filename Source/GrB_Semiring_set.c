//------------------------------------------------------------------------------
// GrB_Semiring_set_*: set a field in a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Semiring_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_set_Scalar
(
    GrB_Semiring semiring,
    GrB_Scalar value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_set_String
(
    GrB_Semiring semiring,
    char * value,
    GrB_Field field
)
{ 
    // FIXME: allow a user-defined semiring to be named
    return ((field == GrB_NAME) ? GrB_ALREADY_SET : GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_set_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_set_ENUM
(
    GrB_Semiring semiring,
    int value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_set_VOID
(
    GrB_Semiring semiring,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

