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

    const char *s ;

    switch ((int) field)
    {

        case GxB_JIT_C_DEFINITION : 
            s = op->defn ;
            break ;

        case GxB_JIT_C_NAME : 
            s = op->name ;
            break ;

        case GrB_NAME : 
            s = GB_op_name_get (op) ;
            break ;

        case GrB_INPUT1TYPE_STRING : 
            s = GB_type_name_get (op->xtype) ;
            break ; ;

        case GrB_INPUT2TYPE_STRING : 
            s = GB_type_name_get (op->ytype) ;
            break ;

        case GrB_OUTPUTTYPE_STRING : 
            s = GB_type_name_get (op->ztype) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    (*value) = ((s == NULL) ? 0 : strlen (s)) + 1 ;

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

