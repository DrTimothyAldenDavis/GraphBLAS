//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineGus_phase2_template:
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// phase2: fine Gustavson task: mask not present, or the mask M(:,j) is
// dense, for C<M>=A*B and C<!M>=A*B.

// The #include'ing file #define's GB_CHECK_MASK_ij, usually as one of:
//      if (Mask [i] != 0) continue ;       // for C<!M>=A*B
//      if (Mask [i] == 0) continue ;       // for C<M>=A*B
//      ;                                   // for C=A*B

{

    // Hf [i] is initially 0.
    // 0 -> 3 : to lock, if i seen for first time
    // 2 -> 3 : to lock, if i seen already
    // 3 -> 2 : to unlock; now i has been seen

    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
    {
        int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
        GB_GET_A_k ;                // get A(:,k)
        if (aknz == 0) continue ;
        GB_GET_B_kj ;               // bkj = B(k,j)
        // scan A(:,k)
        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
        {
            int64_t i = GBI (Ai, pA, avlen) ;  // get A(i,k)

            // check mask condition and skip if C(i,j) is protected by the mask
            GB_CHECK_MASK_ij ;

            GB_MULT_A_ik_B_kj ;      // t = A(i,k) * B(k,j)
            int8_t f ;
            #if GB_IS_ANY_MONOID
            GB_ATOMIC_READ
            f = Hf [i] ;            // grab the entry
            if (f == 2) continue ;  // check if already updated
            GB_ATOMIC_WRITE
            Hf [i] = 2 ;                // flag the entry
            GB_ATOMIC_WRITE_HX (i, t) ;    // Hx [i] = t

            #else

            #if GB_HAS_ATOMIC
            GB_ATOMIC_READ
            f = Hf [i] ;            // grab the entry
            if (f == 2)             // if true, update C(i,j)
            {
                GB_ATOMIC_UPDATE_HX (i, t) ;   // Hx [i] += t
                continue ;          // C(i,j) has been updated
            }
            #endif
            do  // lock the entry
            {
                // do this atomically:
                // { f = Hf [i] ; Hf [i] = 3 ; }
                GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;
            } while (f == 3) ; // lock owner gets f=0 or 2
            if (f == 0)
            { 
                // C(i,j) is a new entry
                GB_ATOMIC_WRITE_HX (i, t) ;    // Hx [i] = t
            }
            else // f == 2
            { 
                // C(i,j) already appears in C(:,j)
                GB_ATOMIC_UPDATE_HX (i, t) ;   // Hx [i] += t
            }
            GB_ATOMIC_WRITE
            Hf [i] = 2 ;                // unlock the entry

            #endif
        }
    }

    // phase2: fine Gustavson task is done -- break out of switch case
    continue ;
}

