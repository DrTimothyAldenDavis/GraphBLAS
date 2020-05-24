//------------------------------------------------------------------------------
// GB_AxB_dot_cij: compute C(i,j) = A(:,i)'*B(:,j)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// computes C(i,j) = A (:,i)'*B(:,j) via sparse dot product.  This template is
// used for all three cases: C=A'*B and C<!M>=A'*B in dot2, and C<M>=A'*B in
// dot3.

// GB_AxB_dot2 defines either one of these, and uses this template twice:

//      GB_PHASE_1_OF_2 ; determine if cij exists, and increment C_count
//      GB_PHASE_2_OF_2 : 2nd phase, compute cij, no realloc of C

// GB_AxB_dot3 defines GB_DOT3, and uses this template just once.

// Only one of the three are #defined: either GB_PHASE_1_OF_2, GB_PHASE_2_OF_2,
// or GB_DOT3.

// When used as the multiplicative operator, the PAIR operator provides some
// useful special cases.  Its output is always one, for any matching pair of
// entries A(k,i)'*B(k,j) for some k.  If the monoid is ANY, then C(i,j)=1 if
// the intersection for the dot product is non-empty.  This intersection has to
// be found, in general.  However, suppose B(:,j) is dense.  Then every entry
// in the pattern of A(:,i)' will produce a 1 from the PAIR operator.  If the
// monoid is ANY, then C(i,j)=1 if A(:,i)' is nonempty.  If the monoid is PLUS,
// then C(i,j) is simply nnz(A(:,i)), assuming no overflow.  The XOR monoid
// acts like a 1-bit summation, so the result of the XOR_PAIR_BOOL semiring
// will be C(i,j) = mod (nnz(A(:,i)'*B(:,j)),2).

// If both A(:,i) and B(:,j) are sparse, then the intersection must still be
// found, so these optimizations can be used only if A(:,i) and/or B(:,j) are
// fully populated.

// For built-in, pre-generated semirings, the PAIR operator is only coupled
// with either the ANY, PLUS, EQ, or XOR monoids, since the other monoids are
// equivalent to the ANY monoid.  With no accumulator, EQ_PAIR is the same as
// ANY_PAIR, they differ for the C+=A'*B operation (see *dot4*).

#include "GB_unused.h"

{

//------------------------------------------------------------------------------
// GB_DOT: cij += A(k,i) * B(k,j) with offset, then break if terminal
//------------------------------------------------------------------------------

// Ai [pA+adelta] and Bi [pB+bdelta] are both equal to the index k.

// use the boolean flag cij_exists to set/check if C(i,j) exists
#define GB_CIJ_CHECK true
#define GB_CIJ_EXISTS (cij_exists)

#if defined ( GB_PHASE_1_OF_2 )

    // symbolic phase (phase 1 of 2 for dot2):
    bool cij_exists = false ;
    #define GB_DOT(adelta,bdelta)                                           \
        cij_exists = true ;                                                 \
        break ;

#else

    // numerical phase (phase 2 of 2 for dot2, or dot3):
    #if GB_IS_PLUS_PAIR_REAL_SEMIRING

        #if GB_CTYPE_IGNORE_OVERFLOW

            // PLUS_PAIR for 64-bit integers, float, and double (not complex):
            // To check if C(i,j) exists, test (cij != 0) when done.  The
            // boolean flag cij_exists is not defined.
            #undef  GB_CIJ_CHECK
            #define GB_CIJ_CHECK false
            #undef  GB_CIJ_EXISTS
            #define GB_CIJ_EXISTS (cij != 0)
            #define GB_DOT(adelta,bdelta)                                   \
                cij++ ;

        #else

            // PLUS_PAIR semiring for small integers
            bool cij_exists = false ;
            #define GB_DOT(adelta,bdelta)                                   \
                cij_exists = true ;                                         \
                cij++ ;

        #endif

    #else

        // all other semirings
        bool cij_exists = false ;
        #define GB_DOT(adelta,bdelta)                                       \
        {                                                                   \
            GB_GETA (aki, Ax, (pA + adelta)) ;  /* aki = A(k,i) */          \
            GB_GETB (bkj, Bx, (pB + bdelta)) ;  /* bkj = B(k,j) */          \
            if (cij_exists)                                                 \
            {                                                               \
                GB_MULTADD (cij, aki, bkj) ;    /* cij += aki * bkj */      \
            }                                                               \
            else                                                            \
            {                                                               \
                /* cij = A(k,i) * B(k,j), and add to the pattern */         \
                cij_exists = true ;                                         \
                GB_MULT (cij, aki, bkj) ;       /* cij = aki * bkj */       \
            }                                                               \
            /* if (cij is terminal) { cij_is_terminal = true ; break ; } */ \
            GB_DOT_TERMINAL (cij) ;                                         \
        }

    #endif

#endif

//------------------------------------------------------------------------------
// C(i,j) = A(:,i)'*B(:,j)
//------------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // get the start of A(:,i) and B(:,j)
    //--------------------------------------------------------------------------

    bool cij_is_terminal = false ;  // C(i,j) not yet reached terminal condition
    int64_t pB = pB_start ;
    int64_t ainz = pA_end - pA ;
    ASSERT (ainz >= 0) ;

    //--------------------------------------------------------------------------
    // declare the cij scalar
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DOT3 )
    GB_CIJ_DECLARE (cij) ;
    #endif

    //--------------------------------------------------------------------------
    // 8 cases for computing C(i,j) = A(:,i)' * B(j,:)
    //--------------------------------------------------------------------------

    if (ainz == 0)
    { 

        //----------------------------------------------------------------------
        // A(:,i) is empty so C(i,j) cannot be present
        //----------------------------------------------------------------------

        ;

    }
    else if (Ai [pA_end-1] < ib_first || ib_last < Ai [pA])
    { 

        //----------------------------------------------------------------------
        // pattern of A(:,i) and B(:,j) do not overlap
        //----------------------------------------------------------------------

        ;

    }
    else if (bjnz == bvlen && ainz == bvlen)
    {

        //----------------------------------------------------------------------
        // both A(:,i) and B(:,j) are dense
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DOT3 )

            #if GB_IS_PAIR_MULTIPLIER

                #if (GB_IS_ANY_MONOID || GB_IS_EQ_MONOID)
                // ANY monoid: take the first entry found
                cij = 1 ;
                #elif (GB_CTYPE_BITS > 0)
                // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
                // for bool, 8-bit, 16-bit, or 32-bit integer
                cij = (GB_CTYPE) (((uint64_t) bvlen) & GB_CTYPE_BITS) ;
                #else
                // PLUS monoid for float, double, or 64-bit integers 
                cij = GB_CTYPE_CAST (bvlen, 0) ;
                #endif

            #else

                // cij = A(0,i) * B(0,j)
                GB_GETA (aki, Ax, pA) ;             // aki = A(0,i)
                GB_GETB (bkj, Bx, pB) ;             // bkj = B(0,j)
                GB_MULT (cij, aki, bkj) ;           // cij = aki * bkj
                GB_PRAGMA_SIMD_DOT (cij)
                for (int64_t k = 1 ; k < bvlen ; k++)
                { 
                    GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
                    GB_MULTADD (cij, aki, bkj) ;        // cij += aki * bkj
                }

            #endif

        #endif

        #if GB_CIJ_CHECK
        cij_exists = true ;
        #endif

    }
    else if (ainz == bvlen)
    {

        //----------------------------------------------------------------------
        // A(:,i) is dense and B(:,j) is sparse
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DOT3 )

            #if GB_IS_PAIR_MULTIPLIER

                #if (GB_IS_ANY_MONOID || GB_IS_EQ_MONOID)
                // ANY monoid: take the first entry found
                cij = 1 ;
                #elif (GB_CTYPE_BITS > 0)
                // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
                // for bool, 8-bit, 16-bit, or 32-bit integer
                cij = (GB_CTYPE) (((uint64_t) bjnz) & GB_CTYPE_BITS) ;
                #else
                // PLUS monoid for float, double, or 64-bit integers 
                cij = GB_CTYPE_CAST (bjnz, 0) ;
                #endif

            #else

                int64_t k = Bi [pB] ;               // first row index of B(:,j)
                // cij = A(k,i) * B(k,j)
                GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
                GB_GETB (bkj, Bx, pB  ) ;           // bkj = B(k,j)
                GB_MULT (cij, aki, bkj) ;           // cij = aki * bkj
                GB_PRAGMA_SIMD_DOT (cij)
                for (int64_t p = pB+1 ; p < pB_end ; p++)
                { 
                    GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                    int64_t k = Bi [p] ;                // next index of B(:,j)
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, p   ) ;           // bkj = B(k,j)
                    GB_MULTADD (cij, aki, bkj) ;        // cij += aki * bkj
                }

            #endif
        #endif

        #if GB_CIJ_CHECK
        cij_exists = true ;
        #endif

    }
    else if (bjnz == bvlen)
    {

        //----------------------------------------------------------------------
        // A(:,i) is sparse and B(:,j) is dense
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DOT3 )

            #if GB_IS_PAIR_MULTIPLIER

                #if (GB_IS_ANY_MONOID || GB_IS_EQ_MONOID)
                // ANY monoid: take the first entry found
                cij = 1 ;
                #elif (GB_CTYPE_BITS > 0)
                // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
                // for bool, 8-bit, 16-bit, or 32-bit integer
                cij = (GB_CTYPE) (((uint64_t) ainz) & GB_CTYPE_BITS) ;
                #else
                // PLUS monoid for float, double, or 64-bit integers 
                cij = GB_CTYPE_CAST (ainz, 0) ;
                #endif

            #else

                int64_t k = Ai [pA] ;               // first row index of A(:,i)
                // cij = A(k,i) * B(k,j)
                GB_GETA (aki, Ax, pA  ) ;           // aki = A(k,i)
                GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
                GB_MULT (cij, aki, bkj) ;           // cij = aki * bkj
                GB_PRAGMA_SIMD_DOT (cij)
                for (int64_t p = pA+1 ; p < pA_end ; p++)
                { 
                    GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                    int64_t k = Ai [p] ;                // next index of A(:,i)
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, p   ) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
                    GB_MULTADD (cij, aki, bkj) ;        // cij += aki * bkj
                }

            #endif

        #endif

        #if GB_CIJ_CHECK
        cij_exists = true ;
        #endif

    }
    else if (ainz > 8 * bjnz)
    {

        //----------------------------------------------------------------------
        // B(:,j) is very sparse compared to A(:,i)
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {
            int64_t ia = Ai [pA] ;
            int64_t ib = Bi [pB] ;
            if (ia < ib)
            { 
                // A(ia,i) appears before B(ib,j)
                // discard all entries A(ia:ib-1,i)
                int64_t pleft = pA + 1 ;
                int64_t pright = pA_end - 1 ;
                GB_TRIM_BINARY_SEARCH (ib, Ai, pleft, pright) ;
                ASSERT (pleft > pA) ;
                pA = pleft ;
            }
            else if (ib < ia)
            { 
                // B(ib,j) appears before A(ia,i)
                pB++ ;
            }
            else // ia == ib == k
            { 
                // A(k,i) and B(k,j) are the next entries to merge
                GB_DOT (0,0) ;
                pA++ ;
                pB++ ;
            }
        }

    }
    else if (bjnz > 8 * ainz)
    {

        //----------------------------------------------------------------------
        // A(:,i) is very sparse compared to B(:,j)
        //----------------------------------------------------------------------

        while (pA < pA_end && pB < pB_end)
        {
            int64_t ia = Ai [pA] ;
            int64_t ib = Bi [pB] ;
            if (ia < ib)
            { 
                // A(ia,i) appears before B(ib,j)
                pA++ ;
            }
            else if (ib < ia)
            { 
                // B(ib,j) appears before A(ia,i)
                // discard all entries B(ib:ia-1,j)
                int64_t pleft = pB + 1 ;
                int64_t pright = pB_end - 1 ;
                GB_TRIM_BINARY_SEARCH (ia, Bi, pleft, pright) ;
                ASSERT (pleft > pB) ;
                pB = pleft ;
            }
            else // ia == ib == k
            { 
                // A(k,i) and B(k,j) are the next entries to merge
                GB_DOT (0,0) ;
                pA++ ;
                pB++ ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A(:,i) and B(:,j) have about the same sparsity
        //----------------------------------------------------------------------

        // load the next 3 entries of Ai [pA ...]
        #define GB_LOAD_A                   \
            a [0] = Ai [pA  ] ;             \
            a [1] = Ai [pA+1] ;             \
            a [2] = Ai [pA+2]

        // load the next 3 entries of Bi [pB ...]
        #define GB_LOAD_B                   \
            b [0] = Bi [pB  ] ;             \
            b [1] = Bi [pB+1] ;             \
            b [2] = Bi [pB+2]

        // munch the next 3 entries of Ai [pA ...] and load the next 3
        #define GB_MUNCH_A                  \
        {                                   \
            pA += 3 ;                       \
            if (pA_end - pA < 3) break ;    \
            GB_LOAD_A ;                     \
            continue ;                      \
        }

        // munch the next 3 entries of Bi [pB ...] and load the next 3
        #define GB_MUNCH_B                  \
        {                                   \
            pB += 3 ;                       \
            if (pB_end - pB < 3) break ;    \
            GB_LOAD_B ;                     \
            continue ;                      \
        }

        // munch the next 3 entries of both Ai and Bi, and load next 3 of both
        #define GB_MUNCH_AB                 \
        {                                   \
            pA += 3 ;                       \
            pB += 3 ;                       \
            if (pB_end - pB < 3) break ;    \
            if (pA_end - pA < 3) break ;    \
            GB_LOAD_A ;                     \
            GB_LOAD_B ;                     \
            continue ;                      \
        }

        // compute the dot product for vectors longer than 3 entries
        if (ainz >= 3 && bjnz >= 3)
        {
            int64_t a [3] ; GB_LOAD_A ;
            int64_t b [3] ; GB_LOAD_B ;
            while (1)
            {
                // get the next 3 entries from each list
                ASSERT (pA_end - pA >= 3) ;
                ASSERT (pB_end - pB >= 3) ;
                if (a [2] < b [0]) GB_MUNCH_A ;
                if (b [2] < a [0]) GB_MUNCH_B ;

                #if GB_IS_PLUS_PAIR_REAL_SEMIRING

                    #if GB_CTYPE_IGNORE_OVERFLOW

                        // no need to handle overflow 
                        cij +=
                        (a [0] == b [0]) + (a [0] == b [1]) + (a [0] == b [2]) +
                        (a [1] == b [0]) + (a [1] == b [1]) + (a [1] == b [2]) +
                        (a [2] == b [0]) + (a [2] == b [1]) + (a [2] == b [2]) ;

                        #if ( GB_PHASE_1_OF_2 )
                        if (cij != 0)
                        {
                            cij_exists = true ;
                            break ;
                        }
                        #endif

                    #else

                        // cij might overflow
                        GB_CTYPE cij_delta =
                        (a [0] == b [0]) + (a [0] == b [1]) + (a [0] == b [2]) +
                        (a [1] == b [0]) + (a [1] == b [1]) + (a [1] == b [2]) +
                        (a [2] == b [0]) + (a [2] == b [1]) + (a [2] == b [2]) ;
                        if (cij_delta !=0)
                        {
                            cij_exists = true ;
                            #if ( GB_PHASE_1_OF_2 )
                            break ;
                            #else
                            cij += cij_delta ;
                            #endif
                        }

                    #endif

                #else

                    int c [3][3] ;

                    c [0][0] = (a [0] == b [0]) ;
                    c [1][0] = (a [1] == b [0]) ;
                    c [2][0] = (a [2] == b [0]) ;

                    c [0][1] = (a [0] == b [1]) ;
                    c [1][1] = (a [1] == b [1]) ;
                    c [2][1] = (a [2] == b [1]) ;

                    c [0][2] = (a [0] == b [2]) ;
                    c [1][2] = (a [1] == b [2]) ;
                    c [2][2] = (a [2] == b [2]) ;

                    if (c [0][0]) { GB_DOT (0, 0) ; }
                    if (c [1][0]) { GB_DOT (1, 0) ; }
                    if (c [2][0]) { GB_DOT (2, 0) ; }

                    if (c [0][1]) { GB_DOT (0, 1) ; }
                    if (c [1][1]) { GB_DOT (1, 1) ; }
                    if (c [2][1]) { GB_DOT (2, 1) ; }

                    if (c [0][2]) { GB_DOT (0, 2) ; }
                    if (c [1][2]) { GB_DOT (1, 2) ; }
                    if (c [2][2]) { GB_DOT (2, 2) ; GB_MUNCH_AB ; }

                #endif

                if (a [2] < b [2]) GB_MUNCH_A else GB_MUNCH_B ;
            }
        }

        #if defined ( GB_PHASE_1_OF_2 )
            // symbolic phase: if C(i,j) already exists, skip the cleanup
            cij_is_terminal = GB_CIJ_EXISTS ;
        #endif
        if (!cij_is_terminal)
        {
            // cleanup for remaining entries of A(:,i) and B(:,j)
            while (pA < pA_end && pB < pB_end)
            {
                int64_t ia = Ai [pA] ;
                int64_t ib = Bi [pB] ;
                if (ia == ib)
                {
                    GB_DOT (0,0) ;
                    pA++ ;
                    pB++ ;
                }
                else
                {
                    pA += (ia < ib) ;
                    pB += (ib < ia) ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // save C(i,j)
    //--------------------------------------------------------------------------

    #if defined ( GB_DOT3 )

        // GB_AxB_dot3: computing C<M>=A'*B
        if (GB_CIJ_EXISTS)
        { 
            // C(i,j) = cij
            GB_CIJ_SAVE (cij, pC) ;
            Ci [pC] = i ;
        }
        else
        { 
            // C(i,j) becomes a zombie
            task_nzombies++ ;
            Ci [pC] = GB_FLIP (i) ;
        }

    #else

        // GB_AxB_dot2: computing C=A'*B or C<!M>=A'*B
        if (GB_CIJ_EXISTS)
        { 
            // C(i,j) = cij
            #if defined ( GB_PHASE_1_OF_2 )
                C_count [Iter_k] ++ ;
            #else
                GB_CIJ_SAVE (cij, cnz) ;
                Ci [cnz++] = i ;
                if (cnz > cnz_last) break ;
            #endif
        }

    #endif
}

#undef GB_DOT
#undef GB_LOAD_A
#undef GB_LOAD_B
#undef GB_MUNCH_A
#undef GB_MUNCH_B
#undef GB_MUNCH_AB
#undef GB_CIJ_EXISTS
#undef GB_CIJ_CHECK

