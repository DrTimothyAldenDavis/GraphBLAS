//------------------------------------------------------------------------------
// GB_sparse_add_M_sparse_setup: C(:,j)<M>=A+B, C sparse/hyper, M sparse, setup
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

    // get M(:,j) if M is sparse or hypersparse
    int64_t pM = -1 ;
    int64_t pM_end = -1 ;

    if (fine_task)
    { 
        // A fine task operates on Mi,Mx [pM...pM_end-1],
        // which is a subset of the vector M(:,j)
        pM     = TaskList [taskid].pM ;
        pM_end = TaskList [taskid].pM_end ;
    }
    else
    {
        int64_t kM = -1 ;
        if (Ch_is_Mh)
        { 
            // Ch is the same as Mh (a deep copy)
            ASSERT (Ch != NULL) ;
            ASSERT (M_is_hyper) ;
            ASSERT (Ch [k] == M->h [k]) ;
            kM = k ;
        }
        else
        { 
            kM = (C_to_M == NULL) ? j : C_to_M [k] ;
        }
        if (kM >= 0)
        { 
            pM     = GBP_M (Mp, kM  , vlen) ;
            pM_end = GBP_M (Mp, kM+1, vlen) ;
        }
    }

    // The "easy mask" condition requires M to be sparse/hyper
    // and structural.  A and B cannot be bitmap.  Also one of
    // the following 3 conditions must hold:
    // (1) all entries are present in A(:,j) and B == M
    // (2) all entries are present in B(:,j) and A == M
    // (3) both A and B are aliased to M
    bool sparse_mask_is_easy =
        Mask_struct &&          // M must be structural
        !A_is_bitmap &&         // A must not be bitmap
        !B_is_bitmap &&         // B must not be bitmap
        ((adense && B == M) ||  // one of 3 conditions holds
         (bdense && A == M) ||
         (A == M && B == M)) ;

    // TODO: add the condition above to GB_add_sparsity,
    // where adense/bdense are true for the whole matrix
    // (adense is true if A is full, or sparse/hypersparse with
    // all entries present).  The test here is done vector by
    // vector, for each A(:,j) and B(:,j).  This is a finer grain
    // test, as compared to a test for all of A and B.

