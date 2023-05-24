//------------------------------------------------------------------------------
// GrB_Monoid_get_*: get a field in a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Monoid_get_Scalar
//------------------------------------------------------------------------------

// FUTURE: add identity and terminal, to replace
// GxB_Monoid_identity and GxB_Monoid_terminal.

GrB_Info GrB_Monoid_get_Scalar
(
    GrB_Monoid monoid,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_Scalar (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_scalar_get ((GB_Operator) (monoid->op), value, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_get_String
(
    GrB_Monoid monoid,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_String (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    const char *name ;

    switch ((int) field)
    {
        case GrB_NAME : 

            name = GB_monoid_name_get (monoid) ;
            if (name == NULL)
            { 
                // construct a name for a user-defined monoid
                sprintf (value, "%s_MONOID", monoid->op->name) ;
            }
            else
            { 
                strcpy (value, name) ;
            }
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GrB_INPUT1TYPE_STRING : 
        case GrB_INPUT2TYPE_STRING : 
        case GrB_OUTPUTTYPE_STRING : 
            return (GB_op_string_get ((GB_Operator) (monoid->op),
                value, field)) ;

        default : ;
            return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GrB_Monoid_get_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_get_ENUM
(
    GrB_Monoid monoid,
    int * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_ENUM (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_enum_get ((GB_Operator) (monoid->op), value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_get_SIZE
(
    GrB_Monoid monoid,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_get_SIZE (monoid, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GrB_NAME : 

            (*value) = GxB_MAX_NAME_LEN ;
            if (monoid->op->opcode == GB_USER_binop_code)
            { 
                (*value) += strlen ("_MONOID") ;
            }
            break ;

        case GrB_INPUT1TYPE_STRING : 
        case GrB_INPUT2TYPE_STRING : 
        case GrB_OUTPUTTYPE_STRING : 
            (*value) = GxB_MAX_NAME_LEN ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_get_VOID
//------------------------------------------------------------------------------

// FUTURE: monoid->op to replace GxB_Monoid_operator.

// FUTURE: can also add identity and terminal, to replace
// GxB_Monoid_identity and GxB_Monoid_terminal.

GrB_Info GrB_Monoid_get_VOID
(
    GrB_Monoid monoid,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

