//------------------------------------------------------------------------------
// GB_subassign_24: make a deep copy of a sparse or dense matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C = A, making a deep copy into an existing non-shallow matrix C, but
// possibly reusing parts of C if C is dense.  See also GB_dup.

// Handles arbitrary typecasting.  A is either sparse or dense; the name of
// the function is a bit of a misnomer since it implies that only the dense
// case is handled.

// FULL: if C sparse and A dense/full, convert C to full

// A can be jumbled, in which case C is also jumbled.

#include "GB_dense.h"
#include "GB_Pending.h"
#define GB_FREE_ALL ;

GrB_Info GB_subassign_24    // C = A, copy A into an existing matrix C
(
    GrB_Matrix C,           // output matrix to modify
    const GrB_Matrix A,     // input matrix to copy
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for C_subassign_24", GB0) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_PENDING_OK (C)) ;

    ASSERT_MATRIX_OK (A, "A for A_subassign_24", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;
    if (A->nvec_nonempty < 0)
    { 
        A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    }

    // the prior pattern of C is discarded
    C->jumbled = false ;

    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // C = A
    //--------------------------------------------------------------------------

    bool copy_dense_A_to_C =            // copy from dense A to dense C if:
        (
            GB_is_dense (C)             // both A and C are dense
            && GB_is_dense (A)
            && !(A->jumbled)            // A cannot be jumbled
            && C->vdim == A->vdim       // A and C have the same size
            && C->vlen == A->vlen
            && C->is_csc == A->is_csc   // A and C have the same format
            && C->x != NULL             // C->x exists
            && !(C->x_shallow)          // C->x is not shallow
        ) ;

    if (copy_dense_A_to_C)
    { 

        //----------------------------------------------------------------------
        // discard the pattern of C
        //----------------------------------------------------------------------

        // make C full, if not full already
        GBURBLE ("(dense copy) ") ;
        C->nzombies = 0 ;                   // overwrite any zombies
        GB_Pending_free (&(C->Pending)) ;   // abandon all pending tuples
        // ensure C is full
        GB_ENSURE_FULL (C) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // copy the pattern from A to C
        //----------------------------------------------------------------------

        // clear prior content of C, but keep the CSR/CSC format and its type
        GBURBLE ("(deep copy) ") ;
        bool C_is_csc = C->is_csc ;
        GB_phbix_free (C) ;
        // copy the pattern, not the values
        GB_OK (GB_dup2 (&C, A, false, C->type, Context)) ;
        C->is_csc = C_is_csc ;      // do not change the CSR/CSC format of C
    }

    //-------------------------------------------------------------------------
    // copy the values from A to C, typecasting as needed
    //-------------------------------------------------------------------------

    if (C->type != A->type)
    { 
        GBURBLE ("(typecast) ") ;
    }

    int64_t anz = GB_NNZ (A) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;
    GB_cast_array (C->x, C->type->code, A->x, A->type->code, A->type->size,
                       anz, nthreads) ;

    //-------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C result for GB_subassign_24", GB0) ;
    return (GrB_SUCCESS) ;
}

