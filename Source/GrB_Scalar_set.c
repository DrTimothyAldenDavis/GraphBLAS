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
    return ((field == GrB_STORAGE_ORIENTATION_HINT) ?
        GrB_SUCCESS : GrB_INVALID_VALUE) ;
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
    return (GrB_NOT_IMPLEMENTED) ;      // FIXME: set the name of a GrB_Scalar
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
    return ((field == GrB_STORAGE_ORIENTATION_HINT) ?
        GrB_SUCCESS : GrB_INVALID_VALUE) ;
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

