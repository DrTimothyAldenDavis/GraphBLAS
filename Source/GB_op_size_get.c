//------------------------------------------------------------------------------
// GB_op_size_get: get a field in an op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_op_size_get
(
    GB_Operator op,
    size_t * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {
        case GrB_NAME : 
        case GrB_INPUT1TYPE_STRING : 
        case GrB_INPUT2TYPE_STRING : 
        case GrB_OUTPUTTYPE_STRING : 
            (*value) = GxB_MAX_NAME_LEN ;
            return (GrB_SUCCESS) ;
        default : ;
            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

