//------------------------------------------------------------------------------
// GB_nvals: number of entries in a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GB_nvals           // get the number of entries in a matrix
(
    GrB_Index *nvals,       // matrix has nvals entries
    const GrB_Matrix A,     // matrix to query
    GB_Context Context
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // delete any lingering zombies and assemble any pending tuples
    // TODO: in v4.0: if zombies but no pending tuples, do not wait,
    // and delete the assertions.
    GrB_Info info ;
    GB_MATRIX_WAIT (A) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    // There are no longer any zombies or pending tuples.  However, except for
    // the side effect (no longer required in the v1.3 spec) of forcing
    // completion in GrB_Matrix_nvals and GrB_Vector_nvals, zombies can be
    // tolerated.

    GB_RETURN_IF_NULL (nvals) ;

    // this has already been done above, but this code must remain in v4.0:
    if (GB_PENDING (A))
    {
        // this will no longer be dead code once the wait is removed above.
        ASSERT (GB_DEAD_CODE) ;
        GB_MATRIX_WAIT (A) ;
    }

    //--------------------------------------------------------------------------
    // return the number of entries in the matrix
    //--------------------------------------------------------------------------

    // Pending tuples are disjoint from the zombies and the live entries in the
    // matrix.  However, there can be duplicates in the pending tuples, and the
    // number of duplicates has not yet been determined.  Thus, zombies can be
    // tolerated but pending tuples cannot.

    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    (*nvals) = GB_NNZ (A) - (A->nzombies) ;
    return (GrB_SUCCESS) ;
}

