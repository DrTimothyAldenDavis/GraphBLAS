//------------------------------------------------------------------------------
// GB_encodify_reduce: encode a GrB_reduce problem, including types and ops
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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
    encoding->suffix_len = 0 ;  // FIXME: use monoid->op->name_len
    uint64_t hash = GB_jitifyer_hash_encoding (encoding) ;
    hash = hash ^ monoid->hash ;

    // FIXME keep track of the strlen of user-defined operator names,
    // so suffix_len can be set above without creating the suffix itself.

    //--------------------------------------------------------------------------
    // monoid and matrix type
    //--------------------------------------------------------------------------

    // FIXME: move this to GB_namify_reduce.

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
        if (monoid->hash != 0)
        {
            // __opname
            (*p++) = '_' ;
            (*p++) = '_' ;
            size_t len1 = strlen (monoid->op->name) ;
            memcpy (p, monoid->op->name, len1) ;
            p += len1 ;
        }

        // terminate the suffix
        (*p) = '\0' ;

        uint32_t len = (uint32_t) (p - suffix) ;
        encoding->suffix_len = len ;
        ASSERT (len == strlen (suffix)) ;
    }

    // return the hash of the problem encoding
    return (hash) ;
}

