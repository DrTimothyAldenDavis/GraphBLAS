//------------------------------------------------------------------------------
// GB_encodify_reduce: encode a GrB_reduce problem, including types and ops
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

uint64_t GB_encodify_reduce // encode a GrB_reduce problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char *suffix,           // suffix for user-defined naming
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A            // input matrix to reduce
)
{

    //--------------------------------------------------------------------------
    // primary encoding
    //--------------------------------------------------------------------------

    bool builtin = GB_enumify_reduce (&encoding->code, monoid, A) ;
    encoding->kcode = 0 ;  // FIXME: GB_JIT_REDUCE_KERNEL
    encoding->suffix_len = 0 ;
    uint64_t hash = GB_jitifyer_encoding_hash (encoding) ;

    //--------------------------------------------------------------------------
    // monoid and matrix type
    //--------------------------------------------------------------------------

    if (builtin)
    {
        // no suffix needed
        (*suffix) = '\0' ;
    }
    else
    {
        // construct the suffix
        char *p = suffix ;

        // append the opname if the monoid is not builtin
        if (!monoid->builtin)
        {
            // __opname
            (*p++) = '_' ;
            (*p++) = '_' ;
            // FIXME keep track of the strlen of operator names
            size_t len1 = strlen (monoid->op->name) ;
            memcpy (p, monoid->op->name, len1) ;
            p += len1 ;
        }

        // append the type of A if it is not builtin
        if (A->type->code == GB_UDT_code)
        {
            // __atypename
            (*p++) = '_' ;
            (*p++) = '_' ;
            size_t len2 = A->type->name_len ;
            ASSERT (len2 == strlen (A->type->name)) ;
            memcpy (p, A->type->name, len2) ;
            p += len2 ;
        }

        // terminate the suffix
        (*p) = '\0' ;

        uint32_t len = (uint32_t) (p - suffix) ;
        encoding->suffix_len = len ;
        ASSERT (len == strlen (suffix)) ;

        // augment the hash with the suffix
        hash = hash ^ GB_jitifyer_suffix_hash (suffix, len) ;
    }

    // return the hash of the problem encoding
    return (hash) ;
}

