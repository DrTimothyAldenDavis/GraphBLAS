//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseHash_phase5:
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase 5: coarse hash task, C=A*B
    //--------------------------------------------------------------------------

    // Initially, H [...].f < mark for all of H [...].f.
    // Let f = H [hash].f and h = H [hash].i

    // f < mark          : unoccupied.
    // h == i, f == mark : occupied with C(i,j)

    for (int64_t kk = kfirst ; kk <= klast ; kk++)
    {
        int64_t pC = Cp [kk] ;      // ok: C is sparse
        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
        if (cjnz == 0) continue ;   // nothing to do
        GB_GET_B_j ;                // get B(:,j)

        #ifdef GB_CHECK_MASK_ij
        GB_GET_M_j                  // get M(:,j)
        #ifndef M_SIZE
        #define M_SIZE 1
        #endif
        const M_TYPE *GB_RESTRICT Mask = ((M_TYPE *) Mx) + (M_SIZE * pM_start) ;
        #else
        if (bjnz == 1)
        { 
            // C(:,j) = A(:,k)*B(k,j), no mask, when nnz (B (:,j)) == 1
            GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE ;
            continue ;
        }
        #endif

        mark++ ;
        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            if (aknz == 0) continue ;
            GB_GET_B_kj ;               // bkj = B(k,j)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)

                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j) is protected by
                // the mask
                GB_CHECK_MASK_ij ;
                #endif

                GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                for (GB_HASH (i))   // find i in hash table
                {
                    if (H [hash].f == mark)
                    {
                        // hash entry is occupied
                        if (H [hash].i == i)
                        { 
                            // i already in the hash table
                            // Hx (hash) += t ;
                            GB_HX_UPDATE (hash, t) ;
                            break ;
                        }
                    }
                    else
                    { 
                        // hash entry is not occupied
                        H [hash].f = mark ;
                        H [hash].i = i ;
                        GB_HX_WRITE (hash, t) ;// Hx (hash) = t
                        Ci [pC++] = i ;        // ok: C sparse
                        break ;
                    }
                }
            }
        }
        // found i if: H [hash].f == mark and H [hash].i == i
        GB_SORT_AND_GATHER_HASHED_C_j (mark, H [hash].i == i)
    }

    continue ;
}

#undef M_TYPE
#undef M_SIZE

