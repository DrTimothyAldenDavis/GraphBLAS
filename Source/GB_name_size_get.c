//------------------------------------------------------------------------------
// GB_name_size_get: get the max size of a name
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_name_size_get (size_t *value, int field)
{
    switch (field)
    {
        case GrB_NAME : 
        case GrB_ELTYPE_STRING : 
            (*value) = GxB_MAX_NAME_LEN ;
            break ;
        default : 
            return (GrB_INVALID_VALUE) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

