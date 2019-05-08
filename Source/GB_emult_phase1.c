//------------------------------------------------------------------------------
// GB_emult_phase1: find # of entries in C=A.*B, C<M>=A.*B, or C<!M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_emult_phase1 counts the number of entries in each vector of C, for
// C=A.*B, C<M>=A.*B, or C<!M>=A.*B, and then does a cumulative sum to find Cp.
// GB_emult_phase1 is preceded by GB_emult_phase0, which finds the non-empty
// vectors of C.  This phase is done entirely in parallel.

// C, M, A, and B can be standard sparse or hypersparse, as determined by
// GB_emult_phase0.  All cases of the mask M are handled: not present, present
// and not complemented, and present and complemented.

// Cp is either freed by GB_emult_phase2, or transplanted into C.

// PARALLEL: done

#include "GB.h"

GrB_Info GB_emult_phase1                // count nnz in each C(:,j)
(
    int64_t **Cp_handle,                // output of size Cnvec+1
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C

    // analysis from GB_emult_phase0
    const int64_t Cnvec,                // # of vectors to compute in C
    const int64_t *restrict Ch,         // Ch is NULL, M->h, A->h, or B->h
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,

    const GrB_Matrix M,         // optional mask, may be NULL
    const bool Mask_comp,       // if true, then M is complemented
    const GrB_Matrix A,         // standard, hypersparse, slice, or hyperslice
    const GrB_Matrix B,         // standard or hypersparse; never a slice
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cp_handle != NULL) ;
    ASSERT (Cnvec_nonempty != NULL) ;
    ASSERT_OK (GB_check (A, "A for emult phase1", GB0)) ;
    ASSERT_OK (GB_check (B, "B for emult phase1", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for emult phase1", GB0)) ;
    ASSERT (A->vdim == B->vdim) ;

    int64_t *restrict Cp = NULL ;
    (*Cp_handle) = NULL ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;
    // TODO reduce nthreads for small problem (work: about O(anz+bnz), but this
    // is a loose upper bound)

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    GB_CALLOC_MEMORY (Cp, GB_IMAX (2, Cnvec+1), sizeof (int64_t)) ;
    if (Cp == NULL)
    { 
        // out of memory
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector of C
    //--------------------------------------------------------------------------

    #define GB_PHASE_1_OF_2
    #include "GB_emult_template.c"

    //--------------------------------------------------------------------------
    // replace Cp with its cumulative sum and return result
    //--------------------------------------------------------------------------

    GB_cumsum (Cp, Cnvec, Cnvec_nonempty, nthreads) ;
    (*Cp_handle) = Cp ;
    return (GrB_SUCCESS) ;
}

