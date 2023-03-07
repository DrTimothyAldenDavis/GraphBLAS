//------------------------------------------------------------------------------
// GB_identity_op: return an identity unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_UnaryOp GB_identity_op (GrB_Type type)
{
    if (type == NULL) return (NULL) ;
    switch (type->code)
    {
        case GB_BOOL_code    : return (GrB_IDENTITY_BOOL  ) ;
        case GB_INT8_code    : return (GrB_IDENTITY_INT8  ) ;
        case GB_INT16_code   : return (GrB_IDENTITY_INT16 ) ;
        case GB_INT32_code   : return (GrB_IDENTITY_INT32 ) ;
        case GB_INT64_code   : return (GrB_IDENTITY_INT64 ) ;
        case GB_UINT8_code   : return (GrB_IDENTITY_UINT8 ) ;
        case GB_UINT16_code  : return (GrB_IDENTITY_UINT16) ;
        case GB_UINT32_code  : return (GrB_IDENTITY_UINT32) ;
        case GB_UINT64_code  : return (GrB_IDENTITY_UINT64) ;
        case GB_FP32_code    : return (GrB_IDENTITY_FP32  ) ;
        case GB_FP64_code    : return (GrB_IDENTITY_FP64  ) ;
        case GB_FC32_code    : return (GxB_IDENTITY_FC32  ) ;
        case GB_FC64_code    : return (GxB_IDENTITY_FC64  ) ;
        default              : ;
    }
    return (NULL) ;
}

