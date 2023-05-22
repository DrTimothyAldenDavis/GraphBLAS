//------------------------------------------------------------------------------
// GB_op_string_get: get a field in an op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_op_string_get
(
    GB_Operator op,
    char * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    GrB_Type type = NULL ;
    char *name ;

    switch ((int) field)
    {
        case GrB_NAME : 

            name = GB_op_name_get (op) ;
            if (name == NULL)
            { 
                return (GrB_INVALID_VALUE) ;
            }
            strcpy (value, name) ;
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GxB_DEFINITION : 

            if (op->defn != NULL)
            { 
                strcpy (value, op->defn) ;
            }
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GrB_INPUT1TYPE_STRING : type = op->xtype ; break ;
        case GrB_INPUT2TYPE_STRING : type = op->ytype ; break ;
        case GrB_OUTPUTTYPE_STRING : type = op->ztype ; break ;
        default : ;
    }

    if (type == NULL)
    {
        return (GrB_INVALID_VALUE) ;
    }

    GB_type_name_get (value, type) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

