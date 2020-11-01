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
// entirely populated.

// For built-in, pre-generated semirings, the PAIR operator is only coupled
// with either the ANY, PLUS, EQ, or XOR monoids, since the other monoids are
// equivalent to the ANY monoid.  With no accumulator, EQ_PAIR is the same as
// ANY_PAIR, they differ for the C+=A'*B operation (see *dot4*).

#include "GB_unused.h"

//------------------------------------------------------------------------------
// GB_DOT: cij += A(k,i) * B(k,j), then break if terminal
//------------------------------------------------------------------------------

// Ai [pA+adelta] and Bi [pB+bdelta] are both equal to the index k.

// use the boolean flag cij_exists to set/check if C(i,j) exists
#undef  GB_CIJ_CHECK
#define GB_CIJ_CHECK true
#undef  GB_CIJ_EXIST
#define GB_CIJ_EXISTS (cij_exists)
#undef  GB_DOT

#if defined ( GB_PHASE_1_OF_2 )

    // symbolic phase (phase 1 of 2 for dot2):
    #define GB_DOT(k,pA,pB)                                                 \
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
            #define GB_DOT(k,pA,pB) cij++ ;

        #else

            // PLUS_PAIR semiring for small integers
            #define GB_DOT(k,pA,pB)                                         \
                cij_exists = true ;                                         \
                cij++ ;

        #endif

    #elif GB_IS_ANY_MONOID

        // ANY monoid
        #define GB_DOT(k,pA,pB)                                             \
        {                                                                   \
            GB_GETA (aki, Ax, pA) ;  /* aki = A(k,i) */                     \
            GB_GETB (bkj, Bx, pB) ;  /* bkj = B(k,j) */                     \
            /* cij = (A')(i,k) * B(k,j), and add to the pattern */          \
            cij_exists = true ;                                             \
            GB_MULT (cij, aki, bkj, i, k, j) ;                              \
            break ;                                                         \
        }

    #else

        // all other semirings
        #define GB_DOT(k,pA,pB)                                             \
        {                                                                   \
            GB_GETA (aki, Ax, pA) ;  /* aki = A(k,i) */                     \
            GB_GETB (bkj, Bx, pB) ;  /* bkj = B(k,j) */                     \
            if (cij_exists)                                                 \
            {                                                               \
                /* cij += (A')(i,k) * B(k,j) */                             \
                GB_MULTADD (cij, aki, bkj, i, k, j) ;                       \
            }                                                               \
            else                                                            \
            {                                                               \
                /* cij = (A')(i,k) * B(k,j), and add to the pattern */      \
                cij_exists = true ;                                         \
                GB_MULT (cij, aki, bkj, i, k, j) ;                          \
            }                                                               \
            /* if (cij is terminal) break ; */                              \
            GB_DOT_TERMINAL (cij) ;                                         \
        }

    #endif

#endif

//------------------------------------------------------------------------------
// C(i,j) = A(:,i)'*B(:,j): a single dot product
//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get the start of A(:,i) and B(:,j)
    //--------------------------------------------------------------------------

    int64_t pB = pB_start ;
    int64_t ainz = pA_end - pA ;
    ASSERT (ainz >= 0) ;
    bool cij_exists = false ;

    //--------------------------------------------------------------------------
    // declare the cij scalar
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DOT3 )
    GB_CIJ_DECLARE (cij) ;
    #endif

    //--------------------------------------------------------------------------
    // 11 cases for computing C(i,j) = A(:,i)' * B(j,:)
    //--------------------------------------------------------------------------

    if (A_is_bitmap && B_is_bitmap)
    {

        //----------------------------------------------------------------------
        // both A and B are bitmap
        //----------------------------------------------------------------------

        for (int64_t k = 0 ; k < vlen ; k++)
        { 
            if (!Ab [pA+k]) continue ;
            if (!Bb [pB+k]) continue ;
            GB_DOT (k, pA+k, pB+k) ;
        }

    }
    else if (A_is_bitmap)
    {

        //----------------------------------------------------------------------
        // A is bitmap; B is sparse, hypersparse, or full
        //----------------------------------------------------------------------

        for (int64_t p = pB ; p < pB_end ; p++)
        { 
            int64_t k = GBI (Bi, p, vlen) ;
            if (!Ab [pA+k]) continue ;
            GB_DOT (k, pA+k, p) ;
        }

    }
    else if (B_is_bitmap)
    {

        //----------------------------------------------------------------------
        // B is bitmap; A is sparse, hypersparse, or full
        //----------------------------------------------------------------------

        for (int64_t p = pA ; p < pA_end ; p++)
        { 
            int64_t k = GBI (Ai, p, vlen) ;
            if (!Bb [pB+k]) continue ;
            GB_DOT (k, p, pB+k) ;
        }

    }
    else if (ainz == 0)
    { 

        //----------------------------------------------------------------------
        // A(:,i) is empty so C(i,j) cannot be present
        //----------------------------------------------------------------------

        ;

    }
    else if (GBI (Ai, pA_end-1, vlen) < ib_first
        || ib_last < GBI (Ai, pA, vlen))
    { 

        //----------------------------------------------------------------------
        // pattern of A(:,i) and B(:,j) do not overlap
        //----------------------------------------------------------------------

        ;

    }
    else if (bjnz == vlen && ainz == vlen)
    {

        //----------------------------------------------------------------------
        // both A(:,i) and B(:,j) are dense
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DOT3 )

            #if GB_IS_PAIR_MULTIPLIER

                #if GB_IS_ANY_MONOID
                // ANY monoid: take the first entry found; this sets cij = 1
                GB_MULT (cij, ignore, ignore, 0, 0, 0) ;
                #elif GB_IS_EQ_MONOID
                // EQ_PAIR semiring: all entries are equal to 1
                cij = 1 ;
                #elif (GB_CTYPE_BITS > 0)
                // PLUS, XOR monoids: A(:,i)'*B(:,j) is nnz(A(:,i)),
                // for bool, 8-bit, 16-bit, or 32-bit integer
                cij = (GB_CTYPE) (((uint64_t) vlen) & GB_CTYPE_BITS) ;
                #else
                // PLUS monoid for float, double, or 64-bit integers 
                cij = GB_CTYPE_CAST (vlen, 0) ;
                #endif

            #else

                // cij = A(0,i) * B(0,j)
                GB_GETA (aki, Ax, pA) ;             // aki = A(0,i)
                GB_GETB (bkj, Bx, pB) ;             // bkj = B(0,j)
                GB_MULT (cij, aki, bkj, i, 0, j) ;  // cij = aki * bkj
                GB_PRAGMA_SIMD_DOT (cij)
                for (int64_t k = 1 ; k < vlen ; k++)
                { 
                    GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
                    GB_MULTADD (cij, aki, bkj, i, k, j) ; // cij += aki * bkj
                }

            #endif

        #endif

        #if GB_CIJ_CHECK
        cij_exists = true ;
        #endif

    }
    else if (ainz == vlen)
    {

        //----------------------------------------------------------------------
        // A(:,i) is dense and B(:,j) is sparse
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DOT3 )

            #if GB_IS_PAIR_MULTIPLIER

                #if GB_IS_ANY_MONOID
                // ANY monoid: take the first entry found; this sets cij = 1
                GB_MULT (cij, ignore, ignore, 0, 0, 0) ;
                #elif GB_IS_EQ_MONOID
                // EQ_PAIR semiring: all entries are equal to 1
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

                // first row index of B(:,j)
                int64_t k = Bi [pB] ;               // ok: B is sparse
                // cij = A(k,i) * B(k,j)
                GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
                GB_GETB (bkj, Bx, pB  ) ;           // bkj = B(k,j)
                GB_MULT (cij, aki, bkj, i, k, j) ;  // cij = aki * bkj
                GB_PRAGMA_SIMD_DOT (cij)
                for (int64_t p = pB+1 ; p < pB_end ; p++)
                { 
                    GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                    // next index of B(:,j)
                    int64_t k = Bi [p] ;                // ok: B is sparse
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, pA+k) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, p   ) ;           // bkj = B(k,j)
                    GB_MULTADD (cij, aki, bkj, i, k, j) ;   // cij += aki * bkj
                }

            #endif
        #endif

        #if GB_CIJ_CHECK
        cij_exists = true ;
        #endif

    }
    else if (bjnz == vlen)
    {

        //----------------------------------------------------------------------
        // A(:,i) is sparse and B(:,j) is dense
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_2_OF_2 ) || defined ( GB_DOT3 )

            #if GB_IS_PAIR_MULTIPLIER

                #if GB_IS_ANY_MONOID
                // ANY monoid: take the first entry found; this sets cij = 1
                GB_MULT (cij, ignore, ignore, 0, 0, 0) ;
                #elif GB_IS_EQ_MONOID
                // EQ_PAIR semiring: all entries are equal to 1
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

                // first row index of A(:,i)
                int64_t k = Ai [pA] ;               // ok: A is sparse
                // cij = A(k,i) * B(k,j)
                GB_GETA (aki, Ax, pA  ) ;           // aki = A(k,i)
                GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
                GB_MULT (cij, aki, bkj, i, k, j) ;  // cij = aki * bkj
                GB_PRAGMA_SIMD_DOT (cij)
                for (int64_t p = pA+1 ; p < pA_end ; p++)
                { 
                    GB_DOT_TERMINAL (cij) ;             // break if cij terminal
                    // next index of A(:,i)
                    int64_t k = Ai [p] ;                // ok: A is sparse
                    // cij += A(k,i) * B(k,j)
                    GB_GETA (aki, Ax, p   ) ;           // aki = A(k,i)
                    GB_GETB (bkj, Bx, pB+k) ;           // bkj = B(k,j)
                    GB_MULTADD (cij, aki, bkj, i, k, j) ;   // cij += aki * bkj
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
            int64_t ia = Ai [pA] ;              // ok: A is sparse
            int64_t ib = Bi [pB] ;              // ok: B is sparse
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
                GB_DOT (ia, pA, pB) ;
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
            int64_t ia = Ai [pA] ;              // ok: A is sparse
            int64_t ib = Bi [pB] ;              // ok: B is sparse
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
                GB_DOT (ia, pA, pB) ;
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

        while (pA < pA_end && pB < pB_end)
        {
            int64_t ia = Ai [pA] ;              // ok: A is sparse
            int64_t ib = Bi [pB] ;              // ok: B is sparse
            if (ia == ib)
            { 
                GB_DOT (ia, pA, pB) ;
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

    //--------------------------------------------------------------------------
    // save C(i,j)
    //--------------------------------------------------------------------------

    #if defined ( GB_DOT3 )

        // GB_AxB_dot3: computing C<M>=A'*B; C and M are sparse/hypersparse
        if (GB_CIJ_EXISTS)
        { 
            // C(i,j) = cij
            GB_PUTC (cij, pC) ;         // Cx [pC] = cij
            Ci [pC] = i ;               // ok: C is sparse
        }
        else
        { 
            // C(i,j) becomes a zombie
            task_nzombies++ ;           // GB_AxB_dot3: computing C<M>=A'*B
            Ci [pC] = GB_FLIP (i) ;     // ok: C is sparse
        }

    #else

        // GB_AxB_dot2: computing C=A'*B, C<M>=A'*B, or C<!M>=A'*B,
        // where M, A, B, and C can have any sparsity pattern

        #if defined ( GB_PHASE_1_OF_2 )

            if (GB_CIJ_EXISTS)
            { 
                // C(i,j) = cij
                C_count [kB]++ ;
            }

        #else

            if (C_is_bitmap)
            { 
                int8_t c = GB_CIJ_EXISTS ;
                if (c) GB_PUTC (cij, cnz) ;
                Cb [cnz] = c ;
                cnvals += c ;
            }
            else if (GB_CIJ_EXISTS)
            { 
                // C(i,j) = cij
                GB_PUTC (cij, cnz) ;    // Cx [cnz] = cij
                Ci [cnz++] = i ;        // ok: C is sparse
                if (cnz > cnz_last) break ;
            }

        #endif
    #endif
}

#undef GB_DOT
#undef GB_CIJ_EXISTS
#undef GB_CIJ_CHECK

