//------------------------------------------------------------------------------
// GrB_Semiring_get_*: get a field in a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Semiring_get_Scalar
//------------------------------------------------------------------------------

// FUTURE: add identity and terminal of the monoid.

GrB_Info GrB_Semiring_get_Scalar
(
    GrB_Semiring semiring,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_Scalar (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_scalar_get ((GB_Operator) (semiring->multiply),
        value, field, Werk)) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_get_String
(
    GrB_Semiring semiring,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_String (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    const char *name ;

    switch ((int) field)
    {
        case GrB_NAME : 

            name = GB_semiring_name_get (semiring) ;
            if (name == NULL)
            { 
                // construct a name for a user-defined semiring
                sprintf (value, "%s_%s_SEMIRING",
                    semiring->add->op->name, semiring->multiply->name) ;
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
            return (GB_op_string_get ((GB_Operator) (semiring->multiply),
                value, field)) ;

        default : ;
            return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GrB_Semiring_get_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_get_ENUM
(
    GrB_Semiring semiring,
    int * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_ENUM (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_enum_get ((GB_Operator) (semiring->multiply), value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_get_SIZE
(
    GrB_Semiring semiring,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_SIZE (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GrB_NAME : 

            (*value) = GxB_MAX_NAME_LEN ;
            if (semiring->add->op->opcode == GB_USER_binop_code ||
                semiring->multiply->opcode == GB_USER_binop_code)
            { 
                (*value) += GxB_MAX_NAME_LEN + strlen ("__SEMIRING") ;
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
// GrB_Semiring_get_VOID
//------------------------------------------------------------------------------

// FUTURE: semiring->add and semiring->multiply to replace GxB_Semiring_add
// and GxB_Semiring_multiply.

// FUTURE: add identity and terminal of the monoid.

GrB_Info GrB_Semiring_get_VOID
(
    GrB_Semiring semiring,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

