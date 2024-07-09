//------------------------------------------------------------------------------
// GB_subassign_26: C(:,j) = A ; append column, no S
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// Method 26: C(:,j) = A ; append column, no S

// M:           NULL
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           matrix
// S:           constructed

// C: hypersparse
// A: sparse

#include "assign/GB_subassign_methods.h"
#include "assign/include/GB_assign_shared_definitions.h"
#undef  GB_FREE_ALL
#define GB_FREE_ALL ;
#define GB_MEM_CHUNK (1024*1024)

GrB_Info GB_subassign_26
(
    GrB_Matrix C,
    // input:
    const int64_t j,        // FUTURE: could handle jlo:jhi
    const GrB_Matrix A,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_HYPERSPARSE (C)) ;
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (!GB_any_aliased (C, A)) ;   // NO ALIAS of C==A
    ASSERT (!GB_PENDING (A)) ;          // FUTURE: could tolerate pending tuples
    ASSERT (!GB_ZOMBIES (A)) ;          // FUTURE: could tolerate zombies
    ASSERT (A->vdim == 1) ;             // FUTURE: could handle A as a matrix
    ASSERT (A->type == C->type) ;       // no typecasting
    ASSERT (!A->iso) ;                  // FUTURE: handle iso case
    ASSERT (!C->iso) ;                  // FUTURE: handle iso case

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    const size_t csize = C->type->size ;
    int64_t cnvec = C->nvec ;
    int64_t cnz = C->nvals ;

    int64_t *restrict Ap = A->p ;
    int64_t *restrict Ai = A->i ;
    GB_void *restrict Ax = (GB_void *) A->x ;
    int64_t anz = A->nvals ;

    //--------------------------------------------------------------------------
    // Method 26: C(:,j) = A ; append column, no S.
    //--------------------------------------------------------------------------

    // Time: Optimal.  Work is O(nnz(A)).

    //--------------------------------------------------------------------------
    // resize C if necessary
    //--------------------------------------------------------------------------

    int64_t cnz_new = cnz + anz ;

    if (cnvec == C->plen)
    {
        // double the size of C->h and C->p if needed
        GB_OK (GB_hyper_realloc (C, GB_IMIN (C->vdim, 2*(C->plen+1)), Werk)) ;
    }

    // printf ("cnz_new %ld nnz max %ld\n", cnz_new, GB_nnz_max (C)) ;

    if (cnz_new > GB_nnz_max (C))
    {
        // double the size of C->i and C->x if needed
        GB_OK (GB_ix_realloc (C, 2*cnz_new + 1)) ;
    }

    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ch = C->h ;
    int64_t *restrict Ci = C->i ;
    GB_void *restrict Cx = (GB_void *) C->x ;

    //--------------------------------------------------------------------------
    // append the new column
    //--------------------------------------------------------------------------

    ASSERT (cnvec == 0 || Ch [cnvec-1] == j-1) ;

    Ch [cnvec] = j ;
    Cp [++(C->nvec)] = cnz_new ;
    C->nvals = cnz_new ;
    if (C->nvec_nonempty >= 0 && anz > 0)
    {
        C->nvec_nonempty++ ;
    }

    C->jumbled = C->jumbled || A->jumbled ;

    // copy the indices and values
    if (anz * (sizeof (int64_t) + csize) <= GB_MEM_CHUNK)
    {
        memcpy (Ci + cnz, Ai, anz * sizeof (int64_t)) ;
        memcpy (Cx + cnz * csize, Ax, anz * csize) ;
    }
    else
    {
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;
        GB_memcpy (Ci + cnz, Ai, anz * sizeof (int64_t), nthreads) ;
        GB_memcpy (Cx + cnz * csize, Ax, anz * csize, nthreads) ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

