//------------------------------------------------------------------------------
// GB_sparse_add_M_easy: C(:,j)<M>=A(:,j)+B(:,j), C sparse/hyper, M easy
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// M is structural, sparse, and "easy".
// A and B are not bitmap.

{

    //--------------------------------------------------------------
    // special case: M is present and very easy to use
    //--------------------------------------------------------------

    //      ------------------------------------------
    //      C      <M> =        A       +       B
    //      ------------------------------------------
    //      sparse  sparse      sparse          sparse
    //      sparse  sparse      sparse          full
    //      sparse  sparse      full            sparse
    //      sparse  sparse      full            full

    // A and B are sparse, hypersparse or full, not bitmap.
    ASSERT (!A_is_bitmap) ;
    ASSERT (!B_is_bitmap) ;
    ASSERT (Mask_struct) ;

    int64_t mjnz = pM_end - pM ;        // nnz (M (:,j))

    #if defined ( GB_PHASE_1_OF_2 )

    // M is structural, and sparse or hypersparse, so every entry
    // in the mask is guaranteed to appear in A+B.  The symbolic
    // count is thus trivial.

    cjnz = mjnz ;

    #else

    // copy the pattern into C (:,j)
    int64_t pC_start = pC ;
    int64_t pM_start = pM ;
    memcpy (Ci + pC, Mi + pM, mjnz * sizeof (int64_t)) ;
    int64_t pA_offset = pA_start - iA_first ;
    int64_t pB_offset = pB_start - iB_first ;

    if (adense && B == M)
    { 

        //----------------------------------------------------------
        // Method11: A dense, B == M
        //----------------------------------------------------------

        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < mjnz ; p++)
        {
            int64_t pM = p + pM_start ;
            int64_t pC = p + pC_start ;
            int64_t i = Mi [pM] ;
            ASSERT (GB_MCAST (Mx, pM, msize)) ;
            ASSERT (GBI_A (Ai, pA_offset + i, vlen) == i) ;
            ASSERT (GBI_B (Bi, pM, vlen) == i) ;
            #ifndef GB_ISO_ADD
            GB_LOAD_A (aij, Ax, pA_offset + i, A_iso) ;
            GB_LOAD_B (bij, Bx, pM, B_iso) ;
            GB_BINOP (GB_CX (pC), aij, bij, i, j) ;
            #endif
        }

    }
    else if (bdense && A == M)
    { 

        //----------------------------------------------------------
        // Method12: B dense, A == M
        //----------------------------------------------------------

        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < mjnz ; p++)
        {
            int64_t pM = p + pM_start ;
            int64_t pC = p + pC_start ;
            int64_t i = Mi [pM] ;
            ASSERT (GB_MCAST (Mx, pM, msize)) ;
            ASSERT (GBI_A (Ai, pM, vlen) == i) ;
            ASSERT (GBI_B (Bi, pB_offset + i, vlen) == i) ;
            #ifndef GB_ISO_ADD
            GB_LOAD_A (aij, Ax, pM, A_iso) ;
            GB_LOAD_B (bij, Bx, pB_offset + i, B_iso) ;
            GB_BINOP (GB_CX (pC), aij, bij, i, j) ;
            #endif
        }

    }
    else // (A == M) && (B == M)
    { 

        //----------------------------------------------------------
        // Method13: A == M == B: all three matrices the same
        //----------------------------------------------------------

        #ifndef GB_ISO_ADD
        GB_PRAGMA_SIMD_VECTORIZE
        for (int64_t p = 0 ; p < mjnz ; p++)
        {
            int64_t pM = p + pM_start ;
            int64_t pC = p + pC_start ;
            #if GB_OP_IS_SECOND
            GB_LOAD_B (t, Bx, pM, B_iso) ;
            #else
            GB_LOAD_A (t, Ax, pM, A_iso) ;
            #endif
            GB_BINOP (GB_CX (pC), t, t, Mi [pM], j) ;
        }
        #endif

    }
    #endif

}
