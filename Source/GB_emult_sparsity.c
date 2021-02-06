//------------------------------------------------------------------------------
// GB_emult_sparsity: determine the sparsity structure for C<M or !M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determines the sparsity structure for C, for computing C=A.*B, C<M>=A.*B,
// or C<!M>=A.*B, based on the sparsity structures of M, A, and B, and whether
// or not M is complemented.  It also decides if the mask M should be applied
// by GB_emult, or if C=A.*B should be computed without the mask, and the mask
// applied later.

// If C should be constructed as hypersparse or sparse, this function simply
// returns GxB_SPARSE.  The final determination is made by GB_emult_phase0.

// If both A and B are full, then GB_ewise calls GB_add instead of GB_emult.
// This is the only case where the eWise multiply can produce a full C matrix,
// and as a result, there is no need for a GB_emult to handle the case when
// C is full.

#include "GB_emult.h"

int GB_emult_sparsity       // return the sparsity structure for C
(
    // output:
    bool *apply_mask,       // if true then mask will be applied by GB_emult
    int *emult_method,      // method to use (0: add, 1: GB_emult_01, etc)
    // input:
    const GrB_Matrix M,     // optional mask for C, unused if NULL
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B      // input B matrix
)
{

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    // Unless deciding otherwise, use the mask if it appears
    (*apply_mask) = (M != NULL) ;
    int C_sparsity ;

    // In the table below, sparse/hypersparse are listed as "sparse".  If C is
    // listed as sparse, it will become sparse or hypersparse, depending on the
    // method.

    bool M_is_sparse_or_hyper = GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M) ;
    bool A_is_sparse_or_hyper = GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A) ;
    bool B_is_sparse_or_hyper = GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B) ;

    bool A_is_full = GB_as_if_full (A) ;
    bool B_is_full = GB_as_if_full (B) ;

    if (M == NULL)
    {

        //      ------------------------------------------
        //      C       =           A       .*      B
        //      ------------------------------------------
        //      sparse  .           sparse          sparse  (method: 99)
        //      sparse  .           sparse          bitmap  (method: 01a)
        //      sparse  .           sparse          full    (method: 01a)
        //      sparse  .           bitmap          sparse  (method: 01b)
        //      bitmap  .           bitmap          bitmap  (method: 18)
        //      bitmap  .           bitmap          full    (method: 18)
        //      sparse  .           full            sparse  (method: 01b)
        //      bitmap  .           full            bitmap  (method: 18)
        //      full    .           full            full    (method: GB_add)

        if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
        {
            // C=A.*B with A and B both sparse/hyper, C sparse
            C_sparsity = GxB_SPARSE ;
            (*emult_method) = GB_EMULT_METHOD_99 ;
        }
        else if (A_is_sparse_or_hyper)
        {
            // C=A.*B with A sparse/hyper, B bitmap/full
            C_sparsity = GxB_SPARSE ;
            (*emult_method) = GB_EMULT_METHOD_01A ;
        }
        else if (B_is_sparse_or_hyper)
        { 
            // C=A.*B with B sparse/hyper, A bitmap/full
            C_sparsity = GxB_SPARSE ;
            (*emult_method) = GB_EMULT_METHOD_01B ;
        }
        else if (A_is_full && B_is_full)
        { 
            // C=A.*B with A and B full, use GB_add
            C_sparsity = GxB_FULL ;
            (*emult_method) = GB_EMULT_METHOD_ADD ;
        }
        else
        { 
            // C=A.*B, otherwise, C bitmap
            C_sparsity = GxB_BITMAP ;
            (*emult_method) = GB_EMULT_METHOD_18 ;
        }

    }
    else if (!Mask_comp)
    {

        if (M_is_sparse_or_hyper)
        { 

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse  (method: 99)
            //      sparse  sparse      sparse          bitmap  (method: 101a)
            //      sparse  sparse      sparse          full    (method: 101a)
            //      sparse  sparse      bitmap          sparse  (method: 101b)
            //      sparse  sparse      bitmap          bitmap  (method: 100)
            //      sparse  sparse      bitmap          full    (method: 100)
            //      sparse  sparse      full            sparse  (method: 101b)
            //      sparse  sparse      full            bitmap  (method: 100)
            //      sparse  sparse      full            full    (method: GB_add)

            // C<M>=A.*B with M sparse/hyper, C sparse
            C_sparsity = GxB_SPARSE ;

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            {
                // C=A.*B with A and B both sparse/hyper, C sparse
                // TODO: check if M should be used now or later
                (*emult_method) = GB_EMULT_METHOD_99 ;
            }
            else if (A_is_sparse_or_hyper)
            {
                // C=A.*B with A sparse/hyper, B bitmap/full
                // TODO: check if M should be used now or later
                (*emult_method) = GB_EMULT_METHOD_101A ;
            }
            else if (B_is_sparse_or_hyper)
            { 
                // C=A.*B with B sparse/hyper, A bitmap/full
                // TODO: check if M should be used now or later
                (*emult_method) = GB_EMULT_METHOD_101B ;
            }
            else if (A_is_full && B_is_full)
            { 
                // C=A.*B with A and B full, use GB_add
                // (*emult_method) = GB_EMULT_METHOD_ADD ;  TODO
                (*emult_method) = GB_EMULT_METHOD_100 ;
            }
            else
            { 
                // C=A.*B, otherwise
                (*emult_method) = GB_EMULT_METHOD_100 ;
            }

        }
        else
        {

            //      ------------------------------------------
            //      C      <M> =        A       .*      B
            //      ------------------------------------------
            //      sparse  bitmap      sparse          sparse  (method: 99)
            //      sparse  bitmap      sparse          bitmap  (method: 01a)
            //      sparse  bitmap      sparse          full    (method: 01a)
            //      sparse  bitmap      bitmap          sparse  (method: 01b)
            //      bitmap  bitmap      bitmap          bitmap  (method: 20)
            //      bitmap  bitmap      bitmap          full    (method: 20)
            //      sparse  bitmap      full            sparse  (method: 01b)
            //      bitmap  bitmap      full            bitmap  (method: 20)
            //      bitmap  bitmap      full            full    (method: GB_add)

            //      ------------------------------------------
            //      C      <M> =        A       .*      B
            //      ------------------------------------------
            //      sparse  full        sparse          sparse  (method: 99)
            //      sparse  full        sparse          bitmap  (method: 01a)
            //      sparse  full        sparse          full    (method: 01a)
            //      sparse  full        bitmap          sparse  (method: 01b)
            //      bitmap  full        bitmap          bitmap  (method: 20)
            //      bitmap  full        bitmap          full    (method: 20)
            //      sparse  full        full            sparse  (method: 01b)
            //      bitmap  full        full            bitmap  (method: 20)
            //      bitmap  full        full            full    (method: GB_add)

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            {
                // C=A.*B with A and B both sparse/hyper, C sparse
                C_sparsity = GxB_SPARSE ;
                (*emult_method) = GB_EMULT_METHOD_99 ;
            }
            else if (A_is_sparse_or_hyper)
            {
                // C=A.*B with A sparse/hyper, B bitmap/full
                C_sparsity = GxB_SPARSE ;
                // (*emult_method) = GB_EMULT_METHOD_01A ;  TODO
                (*emult_method) = GB_EMULT_METHOD_99 ;
            }
            else if (B_is_sparse_or_hyper)
            { 
                // C=A.*B with B sparse/hyper, A bitmap/full
                C_sparsity = GxB_SPARSE ;
                // (*emult_method) = GB_EMULT_METHOD_01B ; TODO
                (*emult_method) = GB_EMULT_METHOD_99 ;
            }
            else if (A_is_full && B_is_full)
            { 
                // C=A.*B with A and B full, use GB_add
                C_sparsity = GxB_BITMAP ;
                // (*emult_method) = GB_EMULT_METHOD_ADD ;  TODO
                (*emult_method) = GB_EMULT_METHOD_20 ;
            }
            else
            { 
                // C=A.*B, otherwise, C bitmap
                C_sparsity = GxB_BITMAP ;
                (*emult_method) = GB_EMULT_METHOD_20 ;
            }
        }

    }
    else // Mask_comp
    {

        if (M_is_sparse_or_hyper)
        {

            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse  (99: M later)
            //      sparse  sparse      sparse          bitmap  (01a: M later)
            //      sparse  sparse      sparse          full    (01a: M later)
            //      sparse  sparse      bitmap          sparse  (01b: M later)
            //      bitmap  sparse      bitmap          bitmap  (method: 19)
            //      bitmap  sparse      bitmap          full    (method: 19)
            //      sparse  sparse      full            sparse  (01b: M later)
            //      bitmap  sparse      full            bitmap  (method: 19)
            //      bitmap  sparse      full            full    (method: GB_add)

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            {
                // C=A.*B with A and B both sparse/hyper, C sparse
                (*apply_mask) = false ;
                C_sparsity = GxB_SPARSE ;
                (*emult_method) = GB_EMULT_METHOD_99 ;
            }
            else if (A_is_sparse_or_hyper)
            {
                // C=A.*B with A sparse/hyper, B bitmap/full
                (*apply_mask) = false ;
                C_sparsity = GxB_SPARSE ;
                (*emult_method) = GB_EMULT_METHOD_01A ;
            }
            else if (B_is_sparse_or_hyper)
            { 
                // C=A.*B with B sparse/hyper, A bitmap/full
                (*apply_mask) = false ;
                C_sparsity = GxB_SPARSE ;
                (*emult_method) = GB_EMULT_METHOD_01B ;
            }
            else if (A_is_full && B_is_full)
            { 
                // C=A.*B with A and B full, use GB_add
                C_sparsity = GxB_BITMAP ;
                // (*emult_method) = GB_EMULT_METHOD_ADD ;  TODO
                (*emult_method) = GB_EMULT_METHOD_19 ;
            }
            else
            { 
                // C=A.*B, otherwise, C bitmap
                C_sparsity = GxB_BITMAP ;
                (*emult_method) = GB_EMULT_METHOD_19 ;
            }

        }
        else
        {

            //      ------------------------------------------
            //      C      <!M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  bitmap      sparse          sparse  (method: 99)
            //      sparse  bitmap      sparse          bitmap  (method: 01a)
            //      sparse  bitmap      sparse          full    (method: 01a)
            //      sparse  bitmap      bitmap          sparse  (method: 01b)
            //      bitmap  bitmap      bitmap          bitmap  (method: 20)
            //      bitmap  bitmap      bitmap          full    (method: 20)
            //      sparse  bitmap      full            sparse  (method: 01b)
            //      bitmap  bitmap      full            bitmap  (method: 20)
            //      bitmap  bitmap      full            full    (method: GB_add)

            //      ------------------------------------------
            //      C      <!M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  full        sparse          sparse  (method: 99)
            //      sparse  full        sparse          bitmap  (method: 01a)
            //      sparse  full        sparse          full    (method: 01a)
            //      sparse  full        bitmap          sparse  (method: 01b)
            //      bitmap  full        bitmap          bitmap  (method: 20)
            //      bitmap  full        bitmap          full    (method: 20)
            //      sparse  full        full            sparse  (method: 01b)
            //      bitmap  full        full            bitmap  (method: 20)
            //      bitmap  full        full            full    (method: GB_add)

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            {
                // C=A.*B with A and B both sparse/hyper, C sparse
                C_sparsity = GxB_SPARSE ;
                (*emult_method) = GB_EMULT_METHOD_99 ;
            }
            else if (A_is_sparse_or_hyper)
            {
                // C=A.*B with A sparse/hyper, B bitmap/full
                C_sparsity = GxB_SPARSE ;
                (*apply_mask) = false ;     // TODO
                (*emult_method) = GB_EMULT_METHOD_01A ;
            }
            else if (B_is_sparse_or_hyper)
            { 
                // C=A.*B with B sparse/hyper, A bitmap/full
                C_sparsity = GxB_SPARSE ;
                (*apply_mask) = false ;     // TODO
                (*emult_method) = GB_EMULT_METHOD_01B ;
            }
            else if (A_is_full && B_is_full)
            { 
                // C=A.*B with A and B full, use GB_add
                C_sparsity = GxB_BITMAP ;
                (*emult_method) = GB_EMULT_METHOD_ADD ;
            }
            else
            { 
                // C=A.*B, otherwise, C bitmap
                C_sparsity = GxB_BITMAP ;
                (*emult_method) = GB_EMULT_METHOD_20 ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (C_sparsity) ;
}

