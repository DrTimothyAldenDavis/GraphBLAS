//------------------------------------------------------------------------------
// GB_enumify2_reduce: enumerate a GrB_reduce problem, including types and ops
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify2_reduce     // enumerate a GrB_reduce problem
(
    // output:
    uint64_t *rcodes,       // unique encoding of the entire problem,
                            // including a fully unique encoding of user-
                            // defined monoids and data types (size 6)
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A            // input matrix to reduce
)
{

    //--------------------------------------------------------------------------
    // primary rcode
    //--------------------------------------------------------------------------

    rcodes [0] = 0 ; // FIXME: enum:  GB_JIT_KERNEL_REDUCE ;
    GB_enumify_reduce (& (rcodes [1]), monoid, A) ;

    //--------------------------------------------------------------------------
    // monoid and matrix type
    //--------------------------------------------------------------------------

    // The codes for the monoid and matrix type are the 64-bit pointers
    // themselves, unless they are builtin objects.  The monoid is builtin if
    // it was pre-defined (GrB_PLUS_FP64_MONOID for example), or if it was
    // constructed by the user application by GrB_Monoid_new from a built-in
    // binary operator.

    // These codes are ephemeral since they are pointers, but that is fine.
    // The hash codes are only used to construct a hash code for the JIT kernel
    // hash table.  That table is empty when GraphBLAS starts (at GrB_init),
    // and previously compiled JIT kernels are loaded and hash later.

    rcodes [2] = (uint64_t) monoid ;
    rcodes [3] = (uint64_t) (A->type) ;
    rcodes [4] = 0 ;
    rcodes [5] = 0 ;
}

