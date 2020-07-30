//------------------------------------------------------------------------------
// GB_AxB_saxpy3_coarseHash_phase1:
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase1: coarse hash task, C=A*B
    //--------------------------------------------------------------------------

    // Initially, H [...].f < mark for all of H [...].f.
    // Let f = H [hash].f and h = H [hash].i

    // f < mark          : unoccupied.
    // h == i, f == mark : occupied with C(i,j)

    for (int64_t kk = kfirst ; kk <= klast ; kk++)
    {
        GB_GET_B_j ;            // get B(:,j)
        if (bjnz == 0)
        { 
            Cp [kk] = 0 ;       // ok: C is sparse
            continue ;
        }

        #ifdef GB_CHECK_MASK_ij
        GB_GET_M_j
        #ifndef M_SIZE
        #define M_SIZE 1
        #endif
        const M_TYPE *GB_RESTRICT Mask = ((M_TYPE *) Mx) + (M_SIZE * pM_start) ;
        #else
        if (bjnz == 1)
        { 
            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
            GB_GET_A_k ;            // get A(:,k)
            Cp [kk] = aknz ;        // nnz(C(:,j)) = nnz(A(:,k))
            continue ;
        }
        #endif

        mark++ ;
        int64_t cjnz = 0 ;
        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
        {
            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
            GB_GET_A_k ;                // get A(:,k)
            // scan A(:,k)
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)

                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j) is protected by
                // the mask
                GB_CHECK_MASK_ij ;
                #endif

                for (GB_HASH (i))       // find i in hash
                {
                    if (H [hash].f == mark)
                    {
                        // position is occupied
                        if (H [hash].i == i)
                        { 
                            // i already in the hash table
                            break ;
                        }
                    }
                    else
                    { 
                        // empty slot found
                        H [hash].f = mark ; // insert C(i,j)
                        H [hash].i = i ;
                        cjnz++ ;  // C(i,j) is a new entry.
                        break ;
                    }
                }
            }
        }
        // count the entries in C(:,j)
        Cp [kk] = cjnz ;        // ok: C is sparse
    }

    continue ;
}

#undef M_TYPE
#undef M_SIZE

