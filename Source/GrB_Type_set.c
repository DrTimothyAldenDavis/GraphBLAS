//------------------------------------------------------------------------------
// GrB_Type_set_*: set a field in a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"
#include <ctype.h>
#include "GB_jitifyer.h"

//------------------------------------------------------------------------------
// GrB_Type_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_Scalar
(
    GrB_Type type,
    GrB_Scalar value,
    GrB_Field field
)
{
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Type_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_String
(
    GrB_Type type,
    char * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Type_set_String (type, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_TYPE_OK (type, "unaryop for get", GB0) ;

    //--------------------------------------------------------------------------
    // quick return for built-in types
    //--------------------------------------------------------------------------

    if (type->code != GB_UDT_code)
    { 
        // built-in type
        return (GrB_ALREADY_SET) ;
    }

    //--------------------------------------------------------------------------
    // set the name or defn of a user-defined type
    //--------------------------------------------------------------------------

    size_t len = strlen (value) ;
    bool compute_hash = false ;

    switch ((int) field)
    {
        case GrB_NAME : 

            if (op->name [0] != '[')    // default name: "[unnamed_user_op]"
            { 
                // name already defined
                return (GrB_ALREADY_SET) ;
            }

            if (value [0] == '[' || len == 0 || len >= GxB_MAX_NAME_LEN)
            { 
                // invalid name: "[" denotes an unnamed user op, the name
                // cannot be empty, and the name cannot exceed
                // GxB_MAX_NAME_LEN-1 characters.
                return (GrB_INVALID_VALUE) ;
            }

            // set the name
            strncpy (op->name, value, GxB_MAX_NAME_LEN-1) ;
            op->name [GxB_MAX_NAME_LEN-1] = '\0' ;
            op->name_len = (int32_t) len ;
            // compute the hash if the op defn has also been set
            compute_hash = (op->defn != NULL) ;
            break ;

        case GxB_DEFINITION : 

            if (op->defn != NULL)
            { 
                // name already defined
                return (GrB_ALREADY_SET) ;
            }

            // allocate space for the definition
            op->defn = GB_MALLOC (len+1, char, &(op->defn_size)) ;
            if (op->defn == NULL)
            { 
                // out of memory
                return (GrB_OUT_OF_MEMORY) ;
            }

            // copy the definition into the new operator
            memcpy (op->defn, value, len+1) ;
            // compute the hash if the op name has also been set
            compute_hash = (op->name [0] != '[') ;
            break ;

        default :  ;
            return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // compute the operator hash, if op->name and op->defn are now both set
    //--------------------------------------------------------------------------

    if (compute_hash)
    { 
        // the op name and defn have been set.  The op can be JIT'd if all its
        // types can be JIT'd; unary ops have no ytype.
        bool jitable =
            (op->ztype->hash != UINT64_MAX) &&
            (op->xtype->hash != UINT64_MAX) &&
            (op->ytype == NULL || op->ytype->hash != UINT64_MAX) ;
        op->hash = GB_jitifyer_hash (op->name, op->name_len, jitable) ;
    }

    return (GrB_SUCCESS) ;
}

}

//------------------------------------------------------------------------------
// GrB_Type_set_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_ENUM
(
    GrB_Type type,
    int value,
    GrB_Field field
)
{
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GrB_Type_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_VOID
(
    GrB_Type type,
    void * value,
    GrB_Field field,
    size_t size
)
{
    return (GrB_NOT_IMPLEMENTED) ;
}

