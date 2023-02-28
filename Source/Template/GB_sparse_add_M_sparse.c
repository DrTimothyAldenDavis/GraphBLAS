//------------------------------------------------------------------------------
// GB_sparse_add_M_sparse: C(:,j)<M>=A(:,j)+B(:,j), C and M sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C and M are both sparse or hyper.

{

    //--------------------------------------------------------------
    // Method14: C and M are sparse or hypersparse
    //--------------------------------------------------------------

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

    // (*) This method is efficient except when either A or B are
    // sparse, and when M is sparse but with many entries.  When M
    // is sparse and either A or B are sparse, the method is
    // designed to be very efficient when M is very sparse compared
    // with A and/or B.  It traverses all entries in the sparse M,
    // and (for sparse A or B) does a binary search for entries in
    // A or B.  In that case, if M has many entries, the mask M
    // should be ignored, and C=A+B should be computed without any
    // mask.  The test for when to use M here should ignore A or B
    // if they are bitmap or full.

    // (+) TODO: if C and M are sparse/hyper, and A and B are
    // both bitmap/full, then use GB_emult_04_template instead,
    // but with (Ab [p] || Bb [p]) instead of (Ab [p] && Bb [p]).

    // A and B can have any sparsity pattern (hypersparse,
    // sparse, bitmap, or full).

    for ( ; pM < pM_end ; pM++)
    {

        //----------------------------------------------------------
        // get M(i,j) for A(i,j) + B (i,j)
        //----------------------------------------------------------

        int64_t i = Mi [pM] ;
        bool mij = GB_MCAST (Mx, pM, msize) ;
        if (!mij) continue ;

        //----------------------------------------------------------
        // get A(i,j)
        //----------------------------------------------------------

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

        //----------------------------------------------------------
        // get B(i,j)
        //----------------------------------------------------------

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

        //----------------------------------------------------------
        // C(i,j) = A(i,j) + B(i,j)
        //----------------------------------------------------------

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
