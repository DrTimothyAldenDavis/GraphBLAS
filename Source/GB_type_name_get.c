//------------------------------------------------------------------------------
// GB_type_name_get: get the name of a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

void GB_type_name_get
(
    char *name,
    GrB_Type type
)
{

    switch (type->code)
    {
        case GB_BOOL_code   : strcpy (name, "GrB_BOOL")   ; break ;
        case GB_INT8_code   : strcpy (name, "GrB_INT8")   ; break ;
        case GB_INT16_code  : strcpy (name, "GrB_INT16")  ; break ;
        case GB_INT32_code  : strcpy (name, "GrB_INT32")  ; break ;
        case GB_INT64_code  : strcpy (name, "GrB_INT64")  ; break ;
        case GB_UINT8_code  : strcpy (name, "GrB_UINT8")  ; break ;
        case GB_UINT16_code : strcpy (name, "GrB_UINT16") ; break ;
        case GB_UINT32_code : strcpy (name, "GrB_UINT32") ; break ;
        case GB_UINT64_code : strcpy (name, "GrB_UINT64") ; break ;
        case GB_FP32_code   : strcpy (name, "GrB_FP32")   ; break ;
        case GB_FP64_code   : strcpy (name, "GrB_FP64")   ; break ;
        case GB_FC32_code   : strcpy (name, "GxB_FC32")   ; break ;
        case GB_FC64_code   : strcpy (name, "GxB_FC64")   ; break ;
        default:
        case GB_UDT_code    : strcpy (name, type->name)   ; break ;
    }
}

