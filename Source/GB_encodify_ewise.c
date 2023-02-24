//------------------------------------------------------------------------------
// GB_encodify_ewise: encode a ewise problem, including types and op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

uint64_t GB_encodify_ewise      // encode an ewise problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const int kcode,            // kernel to encode (add, emult, rowscale, ...)
    const bool C_iso,
    const bool C_in_iso,
    const int C_sparsity,
    const GrB_Type ctype,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const bool flipxy,
    const GrB_Matrix A,
    const GrB_Matrix B
)
{

    //--------------------------------------------------------------------------
    // check if the binaryop is JIT'able
    //--------------------------------------------------------------------------

    if (binaryop != NULL && binaryop->hash == UINT64_MAX)
    {
        // cannot JIT this binaryop
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_enumify_ewise (&encoding->code, C_iso, C_in_iso, C_sparsity, ctype,
        M, Mask_struct, Mask_comp, binaryop, flipxy, A, B) ;
    encoding->kcode = kcode ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    uint64_t hash ;
    if (binaryop == NULL)
    {
        // GrB_wait uses a NULL binaryop; get hash and name from its data type
        ASSERT (A != NULL) ;
        hash = A->type->hash ;
        encoding->suffix_len = (hash == 0) ? 0 : A->type->name_len ;
        (*suffix) = (hash == 0) ? NULL : A->type->name ;
    }
    else
    {
        // typical case: get the hash and name from the binaryop
        hash = binaryop->hash ;
        encoding->suffix_len = (hash == 0) ? 0 : binaryop->name_len ;
        (*suffix) = (hash == 0) ? NULL : binaryop->name ;
    }

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

