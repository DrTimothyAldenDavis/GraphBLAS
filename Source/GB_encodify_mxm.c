//------------------------------------------------------------------------------
// GB_encodify_mxm: encode a GrB_mxm problem, including types and ops
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

uint64_t GB_encodify_mxm        // encode a GrB_mxm problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    int kcode,                  // kernel to encode (dot3, saxpy3, etc)
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Semiring semiring,
    const bool flipxy,
    const GrB_Matrix A,
    const GrB_Matrix B
)
{

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_enumify_mxm (&encoding->code, C->iso, GB_sparsity (C), C->type,
        M, Mask_struct, Mask_comp, semiring, flipxy, A, B) ;
    bool builtin = (semiring->hash == 0) ;
    encoding->kcode = kcode ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    int32_t name_len = semiring->name_len ;
    encoding->suffix_len = (builtin) ? 0 : name_len ;
    (*suffix) = (builtin) ? NULL : semiring->name ;

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    uint64_t hash = GB_jitifyer_hash_encoding (encoding) ;
    hash = hash ^ semiring->hash ;
    return (hash) ;
}

