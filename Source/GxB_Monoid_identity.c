//------------------------------------------------------------------------------
// GxB_Monoid_identity: return the identity of a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// If the monoid has no identity value, a value of zero is returned in the
// identity parameter, but this should never occur since user-visible monoids
// all have identity values.

#include "GB.h"

GrB_Info GxB_Monoid_identity        // return the monoid identity
(
    void *identity,                 // returns the identity of the monoid
    GrB_Monoid monoid               // monoid to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Monoid_identity (&identity, monoid)") ;
    GB_RETURN_IF_NULL (identity) ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    ASSERT_MONOID_OK (monoid, "monoid for identity", GB0) ;

    //--------------------------------------------------------------------------
    // return the identity
    //--------------------------------------------------------------------------

    memset (identity, 0, monoid->op->ztype->size) ;
    if (monoid->identity != NULL)
    { 
        memcpy (identity, monoid->identity, monoid->op->ztype->size) ;
    }
    return (GrB_SUCCESS) ;
}

