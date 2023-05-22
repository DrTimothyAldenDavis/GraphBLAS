//------------------------------------------------------------------------------
// GB_name_get: get a name of a matrix or its type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_name_get (GrB_Matrix A, char *name, int field)
{
    switch (field)
    {
        case GrB_NAME : 
            *name = '\0' ;     // TODO: give GrB_Matrix a name
            break ;
        case GrB_ELTYPE_STRING : 
            GB_type_name_get (name, A->type) ;
            break ;
        default : 
            return (GrB_INVALID_VALUE) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

