//------------------------------------------------------------------------------
// GB_add_template:  phase1 and phase2 for C=A+B, C<M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Computes C=A+B (no mask) or C<M>=A+B (mask present and not complemented).
// C is always sparse or hypersparse.  The complemented case C<!M>=A+B is not
// handled here.  If C is bitmap or full, it is computed elsewhere.

// M can have any sparsity structure:

//      If M is not present, bitmap, or full, then A and B are sparse or
//      hypersparse.  They are not bitmap or full, since in those cases,
//      C will not be sparse/hypersparse, and this method is not used.

//      Otherwise, if M is present and sparse/hypersparse, then A and B can
//      have any sparsity pattern (hyper, sparse, bitmap, or full).

// phase1: does not compute C itself, but just counts the # of entries in each
// vector of C.  Fine tasks compute the # of entries in their slice of a
// single vector of C, and the results are cumsum'd.

// phase2: computes C, using the counts computed by phase1.

{

    //--------------------------------------------------------------------------
    // get A, B, M, and C
    //--------------------------------------------------------------------------

    int taskid ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int8_t  *GB_RESTRICT Ab = A->b ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t vlen = A->vlen ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_full = GB_IS_FULL (A) ;
    int A_nth, A_ntasks ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int8_t  *GB_RESTRICT Bb = B->b ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_full = GB_IS_FULL (B) ;
    int B_nth, B_ntasks ;

    const int64_t *GB_RESTRICT Mp = NULL ;
    const int64_t *GB_RESTRICT Mh = NULL ;
    const int8_t  *GB_RESTRICT Mb = NULL ;
    const int64_t *GB_RESTRICT Mi = NULL ;
    const GB_void *GB_RESTRICT Mx = NULL ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full = GB_IS_FULL (M) ;
    const bool M_is_sparse_or_hyper = M_is_sparse || M_is_hyper ;
    int M_nth, M_ntasks ;
    size_t msize = 0 ;
    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mb = M->b ;
        Mi = M->i ;
        Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
    }

    #if defined ( GB_PHASE_2_OF_2 )
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) A->x ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) B->x ;
    const int64_t  *GB_RESTRICT Cp = C->p ;
    const int64_t  *GB_RESTRICT Ch = C->h ;
          int8_t   *GB_RESTRICT Cb = C->b ;
          int64_t  *GB_RESTRICT Ci = C->i ;
          GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;
    // when C is bitmap or full:
    const int64_t cnz = GB_NNZ_HELD (C) ;
    int C_nth = GB_nthreads (cnz, chunk, nthreads_max) ;
    #endif

// GB_SLICE_MATRIX: Slice the matrix M, A, and B for parallel traversal of a
// single sparse or hypersparse matrix.
#undef  GB_SLICE_MATRIX
#define GB_SLICE_MATRIX(X)                                                     \
{                                                                              \
    X ## _nth = GB_nthreads (GB_NNZ (X) + X->nvec, chunk, nthreads_max) ;      \
    X ## _ntasks = (X ## _nth == 1) ? 1 : (8 * (X ## _nth)) ;                  \
    if (!GB_ek_slice (&(pstart_ ## X ## slice), &(kfirst_ ## X ## slice),      \
        &(klast_ ## X ## slice), X, &(X ## _ntasks)))                          \
    {                                                                          \
        /* out of memory */                                                    \
        GB_FREE_ALL ;                                                          \
        return (GrB_OUT_OF_MEMORY) ;                                           \
    }                                                                          \
}

    //--------------------------------------------------------------------------
    // C=A+B, C<M>=A+B, or C<!M>=A+B: 3 cases for the sparsity of C
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_1_OF_2 )

        // phase1
        #include "GB_add_C_sparse_template.c"

    #else

        // phase2
        if (C_sparsity == GxB_SPARSE || C_sparsity == GxB_HYPERSPARSE)
        {
            // C is sparse or hypersparse (phase1 and phase2)
            #include "GB_add_C_sparse_template.c"
        }
        else if (C_sparsity == GxB_BITMAP)
        {
            // C is bitmap (phase2 only)
            #include "GB_add_C_bitmap_template.c"
        }
        else
        {
            // C is full (phase2 only)
            ASSERT (C_sparsity == GxB_FULL) ;
            #include "GB_add_C_full_template.c"
        }

    #endif
}

