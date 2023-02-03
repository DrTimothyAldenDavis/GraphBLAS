//------------------------------------------------------------------------------
// GB_Semiring_new: create a new semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The semiring struct is already allocated on input.

#include "GB.h"
#include "GB_Semiring_new.h"
#include "GB_jitifyer.h"

GrB_Info GB_Semiring_new            // create a semiring
(
    GrB_Semiring semiring,          // semiring to create
    GrB_Monoid add,                 // additive monoid of the semiring
    GrB_BinaryOp multiply           // multiply operator of the semiring
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (semiring != NULL) ;
    ASSERT (add != NULL) ;
    ASSERT (multiply != NULL) ;
    ASSERT_MONOID_OK (add, "semiring->add", GB0) ;
    ASSERT_BINARYOP_OK (multiply, "semiring->multiply", GB0) ;

    //--------------------------------------------------------------------------
    // create the semiring
    //--------------------------------------------------------------------------

    // z = multiply(x,y); type of z must match monoid z = add(z,z)
    if (multiply->ztype != add->op->ztype)
    {
        return (GrB_DOMAIN_MISMATCH) ;
    }

    // initialize the semiring
    semiring->magic = GB_MAGIC ;
    semiring->add = add ;
    semiring->multiply = multiply ;
    if (semiring->add->hash == 0 && semiring->multiply->hash == 0)
    {
        // semiring consists of builtin types and operators only
        semiring->hash = 0 ;
    }
    else
    {
        // construct the semiring hash from the monoid and mult binop hashes
        uint64_t hashes [2] ;
        hashes [0] = semiring->add->hash ;
        hashes [1] = semiring->multiply->hash ;
        semiring->hash = GB_jitifyer_hash (hashes, 2*sizeof (uint64_t)) ;
    }
    ASSERT_SEMIRING_OK (semiring, "new semiring", GB0) ;
    return (GrB_SUCCESS) ;
}

