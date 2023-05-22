//------------------------------------------------------------------------------
// GB_op_string_set: set the name or defn of an operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"
#include <ctype.h>
#include "GB_jitifyer.h"

GrB_Info GB_op_string_set
(
    GB_Operator op,
    char * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // quick return for built-in operators
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    if (!((opcode == GB_USER_unop_code) || (opcode == GB_USER_idxunop_code) ||
       (opcode == GB_USER_binop_code)))
    { 
        // built-in operator
        return (GrB_ALREADY_SET) ;
    }

    //--------------------------------------------------------------------------
    // set the name or defn of a user-defined op
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

