//------------------------------------------------------------------------------
// GB_AxB_saxpy3_symbolic: symbolic analysis for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Symbolic analysis for C=A*B, C<M>=A*B or C<!M>=A*B, via GB_AxB_saxpy3.
// Coarse tasks compute nnz (C (:,j)) for each of their vectors j.  Fine tasks
// just scatter the mask M into the hash table.  This phase does not depend on
// the semiring, nor does it depend on the type of C, A, or B.  It does access
// the values of M, if the mask matrix M is present and not structural.

#include "GB_AxB_saxpy3.h"
#include "GB_AxB_saxpy3_template.h"
#include "GB_atomics.h"
#include "GB_bracket.h"
// GB_GET_A_k and GB_GET_M_j declare aknz and mjnz, but these are unused here.
#include "GB_unused.h"

void GB_AxB_saxpy3_symbolic
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    bool Mask_comp,             // M complemented, or not
    bool Mask_struct,           // M structural, or not
    bool M_dense_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *TaskList,     // list of tasks, and workspace
    int ntasks,                 // total number of tasks
    int nfine,                  // number of fine tasks
    int nthreads                // number of threads
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_PENDING (M)) ; 

    ASSERT (!GB_ZOMBIES (A)) ; 
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ; 

    ASSERT (!GB_ZOMBIES (B)) ; 
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_PENDING (B)) ; 

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Cp = C->p ;
    const int64_t cvlen = C->vlen ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const int64_t bvlen = B->vlen ;
    const bool B_jumbled = B->jumbled ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const bool A_is_hyper = GB_IS_HYPER (A) ;
    const bool A_jumbled = A->jumbled ;

    const int64_t *GB_RESTRICT Mp = NULL ;
    const int64_t *GB_RESTRICT Mh = NULL ;
    const int64_t *GB_RESTRICT Mi = NULL ;
    const GB_void *GB_RESTRICT Mx = NULL ;
    size_t msize = 0 ;
    int64_t mnvec = 0 ;
    int64_t mvlen = 0 ;
    bool M_is_hyper = false ;
    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mi = M->i ;
        Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
        mnvec = M->nvec ;
        mvlen = M->vlen ;
        M_is_hyper = (Mh != NULL) ;
    }

    // 3 cases:
    //      M not present and Mask_comp false: compute C=A*B
    //      M present     and Mask_comp false: compute C<M>=A*B
    //      M present     and Mask_comp true : compute C<!M>=A*B
    // If M is NULL on input, then Mask_comp is also false on input.

    bool mask_is_M = (M != NULL && !Mask_comp) ;

    // Ignore the mask if not complemented, dense and used in-place, and
    // structural (and thus the same as no mask at all).  For this case,
    // all-hash tasks are used (see M_dense_in_place computation in
    // GB_AxB_saxpy3).
    bool ignore_mask = (mask_is_M && M_dense_in_place && Mask_struct) ;

    //==========================================================================
    // phase1: count nnz(C(:,j)) for coarse tasks, scatter M for fine tasks
    //==========================================================================

    // At this point, all of H [...].f is zero, for all tasks.
    // H [...].i and H [...].x are not initialized.

    // phase1 does not depend on the type or values of C, but the hash table
    // data structure does, so a switch is necessary.  Since H [...].x is not
    // accessed, some of the methods are combined to reduce code size.

    switch (C->type->code)
    {
        case GB_BOOL_code   :
        case GB_INT8_code   :
        case GB_UINT8_code  :
            #define GB_HASH_FINEGUS GB_hash_fineGus_uint8_t
            #define GB_HASH_TYPE    GB_hash_uint8_t
            #define GB_HASH_COARSE  GB_hash_coarse_uint8_t
            #include "GB_AxB_saxpy3_symbolic_template.c"

        case GB_INT16_code  :
        case GB_UINT16_code :
            #define GB_HASH_FINEGUS GB_hash_fineGus_uint16_t
            #define GB_HASH_TYPE    GB_hash_uint16_t
            #define GB_HASH_COARSE  GB_hash_coarse_uint16_t
            #include "GB_AxB_saxpy3_symbolic_template.c"

        case GB_INT32_code  :
        case GB_UINT32_code :
        case GB_FP32_code   :
            #define GB_HASH_FINEGUS GB_hash_fineGus_uint32_t
            #define GB_HASH_TYPE    GB_hash_uint32_t
            #define GB_HASH_COARSE  GB_hash_coarse_uint32_t
            #include "GB_AxB_saxpy3_symbolic_template.c"

        case GB_INT64_code  :
        case GB_UINT64_code :
        case GB_FP64_code   :
        case GB_FC32_code   :
            #define GB_HASH_FINEGUS GB_hash_fineGus_uint64_t
            #define GB_HASH_TYPE    GB_hash_uint64_t
            #define GB_HASH_COARSE  GB_hash_coarse_uint64_t
            #include "GB_AxB_saxpy3_symbolic_template.c"

        case GB_FC64_code   :
            #define GB_HASH_FINEGUS GB_hash_fineGus_GxB_FC64_t
            #define GB_HASH_TYPE    GB_hash_GxB_FC64_t
            #define GB_HASH_COARSE  GB_hash_coarse_GxB_FC64_t
            #include "GB_AxB_saxpy3_symbolic_template.c"

        default             :
            // C->type == op->ztype is a user-defined type
            #define GB_HASH_FINEGUS GB_hash_fineGus_GB_void
            #define GB_HASH_TYPE    GB_hash_GB_void
            #define GB_HASH_COARSE  GB_hash_coarse_GB_void
            #include "GB_AxB_saxpy3_symbolic_template.c"
    }
}

