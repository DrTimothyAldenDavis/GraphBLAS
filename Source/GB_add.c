//------------------------------------------------------------------------------
// GB_add: C = A+B, C<M>=A+B, or C<!M> = A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_add (C, M, A, B, op), does C<M>=op(A,B), using the given operator
// element-wise on the matrices A and B.  The result is typecasted as needed.
// The pattern of C is the union of the pattern of A and B, intersection with
// the mask M or !M, if present.

// Let the op be z=f(x,y) where x, y, and z have type xtype, ytype, and ztype.
// If both A(i,j) and B(i,j) are present, then:

//      C(i,j) = (ctype) op ((xtype) A(i,j), (ytype) B(i,j))

// If just A(i,j) is present but not B(i,j), then:

//      C(i,j) = (ctype) A (i,j)

// If just B(i,j) is present but not A(i,j), then:

//      C(i,j) = (ctype) B (i,j)

// ctype is the type of matrix C.  The pattern of C is the union of A and B.

// op may be NULL.  In this case, the intersection of A and B must be empty.
// This is used by GB_wait only, for merging the pending tuple matrix T into A.
// Any duplicate pending tuples have already been summed in T, so the
// intersection of T and A is always empty.

// PARALLEL: done, except for phase0 when both A and B are hypersparse, and
// phase2 to prune empty vectors from C->h.  Consider a single phase method
// when nthreads == 1.

#include "GB.h"

GrB_Info GB_add             // C=A+B, C<M>=A+B, or C<!M>=A+B
(
    GrB_Matrix *Chandle,    // output matrix (unallocated on input)
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask for C, unused if NULL
    const bool Mask_comp,   // descriptor for M
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B,     // input B matrix
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Chandle != NULL) ;
    ASSERT_OK (GB_check (A, "A for add phased", GB0)) ;
    ASSERT_OK (GB_check (B, "B for add phased", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (op, "op for add phased", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for add phased", GB0)) ;
    ASSERT (A->vdim == B->vdim && A->vlen == B->vlen) ;
    if (M != NULL)
    { 
        ASSERT (A->vdim == M->vdim && A->vlen == M->vlen) ;
    }

    // delete any lingering zombies and assemble any pending tuples
    GB_WAIT (M) ;
    GB_WAIT (A) ;
    GB_WAIT (B) ;

    //--------------------------------------------------------------------------
    // phase0: determine the vectors in C(:,j)
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    int64_t Cnvec, max_Cnvec, Cnvec_nonempty ;
    int64_t *Cp = NULL, *Ch = NULL, *C_to_A = NULL, *C_to_B = NULL ;
    bool Ch_is_Mh ;

    GrB_Info info = GB_add_phase0 (
        &Cnvec, &max_Cnvec, &Ch, &C_to_A, &C_to_B, &Ch_is_Mh, // comp. by phase0
        M, Mask_comp, A, B, Context) ;          // original input
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // phase1: count the number of entries in each vector of C
    //--------------------------------------------------------------------------

    info = GB_add_phase1 (
        &Cp, &Cnvec_nonempty,                   // computed by phase1
        op == NULL,                             // if true, A and B disjoint
        Cnvec, Ch, C_to_A, C_to_B, Ch_is_Mh,    // from phase0
        M, Mask_comp, A, B, Context) ;          // original input
    if (info != GrB_SUCCESS)
    { 
        // out of memory; free everything allocated by GB_add_phase0
        GB_FREE_MEMORY (Ch,     max_Cnvec, sizeof (int64_t)) ;
        GB_FREE_MEMORY (C_to_A, max_Cnvec, sizeof (int64_t)) ;
        GB_FREE_MEMORY (C_to_B, max_Cnvec, sizeof (int64_t)) ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // phase2: compute the entries (indices and values) in each vector of C
    //--------------------------------------------------------------------------

    // Cp and Ch are either freed by phase2, or transplanted into C.
    // Either way, they are not freed here.

    info = GB_add_phase2 (
        &C, ctype, C_is_csc, op,                // computed or used by phase2
        Cp, Cnvec_nonempty,                             // from phase1
        Cnvec, max_Cnvec, Ch, C_to_A, C_to_B, Ch_is_Mh, // from phase0
        M, Mask_comp, A, B, Context) ;                  // original input

    // free workspace
    GB_FREE_MEMORY (C_to_A, max_Cnvec, sizeof (int64_t)) ;
    GB_FREE_MEMORY (C_to_B, max_Cnvec, sizeof (int64_t)) ;

    if (info != GrB_SUCCESS)
    { 
        // out of memory; note that Cp and Ch are already freed
        return (info) ;
    }

    // if successful, Ch and Cp must not be freed; they are now C->h and C->p

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_OK (GB_check (C, "C output for add phased", GB0)) ;
    (*Chandle) = C ;
    return (GrB_SUCCESS) ;
}

