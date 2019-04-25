//------------------------------------------------------------------------------
// GB_AxB_dot2_phase1: count entries in C=A'*B, C<M>=A'*B, or C<!M>=A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Count the number of entries in each vector of C, for C=A'B, C<M>=A'*B, or
// C<!M>=A'*B.

#include "GB.h"

GrB_Info GB_AxB_dot2_phase1         // C<M> = A'*B, dot product counts
(
    int64_t **C_count_handle,       // output of size B->nvec
    const GrB_Matrix M,             // mask matrix for C<M>=A'*B or C<!M>=A'*B
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,             // input matrix, may be a slice
    const GrB_Matrix B,             // input matrix
    int nthreads,
    int naslice,
    int nbslice
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_Context Context = NULL ;
    ASSERT (C_count_handle != NULL) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for dot2 phase1 A'*B", GB0)) ;
    ASSERT_OK (GB_check (A, "A for dot2 phase1 A'*B", GB0)) ;
    ASSERT_OK (GB_check (B, "B for dot2 phase1 A'*B", GB0)) ;
    ASSERT (!GB_PENDING (M)) ; ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (!GB_PENDING (A)) ; ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (B)) ; ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (A->vlen == B->vlen) ;
    ASSERT (A->nvec_nonempty == GB_nvec_nonempty (A, NULL)) ;

    //--------------------------------------------------------------------------
    // allocate result
    //--------------------------------------------------------------------------

    (*C_count_handle) = NULL ;
    int64_t *restrict C_count = NULL ;
    GB_CALLOC_MEMORY (C_count, B->nvec, sizeof (int64_t)) ;
    if (C_count == NULL)
    {
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // compute the column counts
    //--------------------------------------------------------------------------

    #define GB_PHASE_1_OF_2
    #include "GB_AxB_dot_meta.c"

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

//  for (int64_t k = 0 ; k < B->nvec ; k++)
//      printf ("C_count ["GBd"] = "GBd"\n", k, C_count [k]) ;

    (*C_count_handle) = C_count ;
    return (GrB_SUCCESS) ;
}

