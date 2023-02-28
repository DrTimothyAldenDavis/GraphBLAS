//------------------------------------------------------------------------------
// GB_add_sparse_M_sparse: C(:,j)<M>=A(:,j)+B(:,j), C and M sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C and M are both sparse or hyper.

{

    //--------------------------------------------------------------------------
    // setup for C(:,j)<M> = A(:,j) + B(:,j), where C and M are sparse/hyper
    //--------------------------------------------------------------------------

    // get M(:,j) where M is sparse or hypersparse
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

    // The "easy mask" condition requires M to be sparse/hyper and structural.
    // A and B cannot be bitmap, and one of these 3 conditions must hold:
    // (1) all entries are present in A(:,j) and B == M
    // (2) all entries are present in B(:,j) and A == M
    // (3) both A and B are aliased to M

    if (Mask_struct &&          // M must be structural
        !A_is_bitmap &&         // A must not be bitmap
        !B_is_bitmap &&         // B must not be bitmap
        ((adense && B == M) ||  // one of 3 conditions holds
         (bdense && A == M) ||
         (A == M && B == M)))
    {

        //----------------------------------------------------------------------
        // special case: M is present and very easy to use
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C      <M> =        A       +       B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse
        //      sparse  sparse      sparse          full
        //      sparse  sparse      full            sparse
        //      sparse  sparse      full            full

        // A and B are sparse, hypersparse or full, not bitmap.

        int64_t mjnz = pM_end - pM ;        // nnz (M (:,j))

        #if defined ( GB_PHASE_1_OF_2 )

        // M is structural, and sparse or hypersparse, so every entry in the
        // mask is guaranteed to appear in A+B.  The symbolic count is thus
        // trivial.

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

            //------------------------------------------------------------------
            // Method11: A dense, B == M
            //------------------------------------------------------------------

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

            //------------------------------------------------------------------
            // Method12: B dense, A == M
            //------------------------------------------------------------------

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

            //------------------------------------------------------------------
            // Method13: A == M == B: all three matrices the same
            //------------------------------------------------------------------

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
    else
    {

        //----------------------------------------------------------------------
        // Method14: C and M are sparse or hypersparse
        //----------------------------------------------------------------------

        //      ------------------------------------------
        //      C      <M> =        A       +       B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse  (*)
        //      sparse  sparse      sparse          bitmap  (*)
        //      sparse  sparse      sparse          full    (*)
        //      sparse  sparse      bitmap          sparse  (*)
        //      sparse  sparse      bitmap          bitmap  (+)
        //      sparse  sparse      bitmap          full    (+)
        //      sparse  sparse      full            sparse  (*)
        //      sparse  sparse      full            bitmap  (+)
        //      sparse  sparse      full            full    (+)

        // (*) This method is efficient except when either A or B are sparse,
        // and when M is sparse but with many entries.  When M is sparse and
        // either A or B are sparse, the method is designed to be very
        // efficient when M is very sparse compared with A and/or B.  It
        // traverses all entries in the sparse M, and (for sparse A or B) does
        // a binary search for entries in A or B.  In that case, if M has many
        // entries, the mask M should be ignored, and C=A+B should be computed
        // without any mask.  The test for when to use M here should ignore A
        // or B if they are bitmap or full.

        // (+) TODO: if C and M are sparse/hyper, and A and B are both
        // bitmap/full, then use GB_emult_04_template instead, but with (Ab [p]
        // || Bb [p]) instead of (Ab [p] && Bb [p]).

        // A and B can have any sparsity pattern (hypersparse, sparse, bitmap,
        // or full).

        for ( ; pM < pM_end ; pM++)
        {

            //------------------------------------------------------------------
            // get M(i,j) for A(i,j) + B (i,j)
            //------------------------------------------------------------------

            int64_t i = Mi [pM] ;
            bool mij = GB_MCAST (Mx, pM, msize) ;
            if (!mij) continue ;

            //------------------------------------------------------------------
            // get A(i,j)
            //------------------------------------------------------------------

            bool afound ;
            if (adense)
            { 
                // A is dense, bitmap, or full; use quick lookup
                pA = pA_start + (i - iA_first) ;
                afound = GBB_A (Ab, pA) ;
            }
            else if (A == M)
            { 
                // A is aliased to M
                pA = pM ;
                afound = true ;
            }
            else
            { 
                // A is sparse; use binary search.  This is slow unless
                // M is very sparse compared with A.
                int64_t apright = pA_end - 1 ;
                GB_BINARY_SEARCH (i, Ai, pA, apright, afound) ;
            }

            ASSERT (GB_IMPLIES (afound, GBI_A (Ai, pA, vlen) == i)) ;

            //------------------------------------------------------------------
            // get B(i,j)
            //------------------------------------------------------------------

            bool bfound ;
            if (bdense)
            { 
                // B is dense; use quick lookup
                pB = pB_start + (i - iB_first) ;
                bfound = GBB_B (Bb, pB) ;
            }
            else if (B == M)
            { 
                // B is aliased to M
                pB = pM ;
                bfound = true ;
            }
            else
            { 
                // B is sparse; use binary search.  This is slow unless
                // M is very sparse compared with B.
                int64_t bpright = pB_end - 1 ;
                GB_BINARY_SEARCH (i, Bi, pB, bpright, bfound) ;
            }

            ASSERT (GB_IMPLIES (bfound, GBI_B (Bi, pB, vlen) == i)) ;

            //------------------------------------------------------------------
            // C(i,j) = A(i,j) + B(i,j)
            //------------------------------------------------------------------

            if (afound && bfound)
            { 
                // C (i,j) = A (i,j) + B (i,j)
                #if defined ( GB_PHASE_1_OF_2 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                GB_LOAD_A (aij, Ax, pA, A_iso) ;
                GB_LOAD_B (bij, Bx, pB, B_iso) ;
                GB_BINOP (GB_CX (pC), aij, bij, i, j) ;
                #endif
                pC++ ;
                #endif
            }
            else if (afound)
            { 
                #if defined ( GB_PHASE_1_OF_2 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = A(i,j) + beta
                    GB_LOAD_A (aij, Ax, pA, A_iso) ;
                    GB_BINOP (GB_CX (pC), aij, beta_scalar, i, j) ;
                }
                #else
                { 
                    // C (i,j) = A (i,j)
                    GB_COPY_A_TO_C (Cx, pC, Ax, pA, A_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
            else if (bfound)
            { 
                #if defined ( GB_PHASE_1_OF_2 )
                cjnz++ ;
                #else
                Ci [pC] = i ;
                #ifndef GB_ISO_ADD
                #if GB_IS_EWISEUNION
                { 
                    // C (i,j) = alpha + B(i,j)
                    GB_LOAD_B (bij, Bx, pB, B_iso) ;
                    GB_BINOP (GB_CX (pC), alpha_scalar, bij, i, j) ;
                }
                #else
                { 
                    // C (i,j) = B (i,j)
                    GB_COPY_B_TO_C (Cx, pC, Bx, pB, B_iso) ;
                }
                #endif
                #endif
                pC++ ;
                #endif
            }
        }

        #if defined ( GB_PHASE_2_OF_2 )
        ASSERT (pC == pC_end) ;
        #endif
    }
}

