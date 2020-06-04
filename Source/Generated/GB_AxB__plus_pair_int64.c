
//------------------------------------------------------------------------------
// GB_AxB:  hard-coded functions for semiring: C<M>=A*B or A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated).

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "GB_ek_slice.h"
#include "GB_bracket.h"
#include "GB_iterator.h"
#include "GB_sort.h"
#include "GB_atomics.h"
#include "GB_AxB_saxpy3.h"
#include "GB_AxB__include.h"

// The C=A*B semiring is defined by the following types and operators:

// A'*B function (dot2):     GB_Adot2B__plus_pair_int64
// A'*B function (dot3):     GB_Adot3B__plus_pair_int64
// C+=A'*B function (dot4):  GB_Adot4B__plus_pair_int64
// A*B function (saxpy3):    GB_Asaxpy3B__plus_pair_int64

// C type:   int64_t
// A type:   int64_t
// B type:   int64_t

// Multiply: z = 1
// Add:      cij += z
//           'any' monoid?  0
//           atomic?        1
//           OpenMP atomic? 1
// MultAdd:  cij += 1
// Identity: 0
// Terminal: ;

#define GB_ATYPE \
    int64_t

#define GB_BTYPE \
    int64_t

#define GB_CTYPE \
    int64_t

// true for int64, uint64, float, double, float complex, and double complex 
#define GB_CTYPE_IGNORE_OVERFLOW \
    1

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA) \
    ;

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB) \
    ;

#define GB_CX(p) Cx [p]

// multiply operator
#define GB_MULT(z, x, y) \
    z = 1

// cast from a real scalar (or 2, if C is complex) to the type of C
#define GB_CTYPE_CAST(x,y) \
    ((int64_t) x)

// multiply-add
#define GB_MULTADD(z, x, y) \
    z += 1

// monoid identity value
#define GB_IDENTITY \
    0

// break if cij reaches the terminal value (dot product only)
#define GB_DOT_TERMINAL(cij) \
    ;

// simd pragma for dot-product loop vectorization
#define GB_PRAGMA_SIMD_DOT(cij) \
    GB_PRAGMA_SIMD_REDUCTION (+,cij)

// simd pragma for other loop vectorization
#define GB_PRAGMA_SIMD_VECTORIZE GB_PRAGMA_SIMD

// 1 for the PLUS_PAIR_(real) semirings, not for the complex case
#define GB_IS_PLUS_PAIR_REAL_SEMIRING \
    1

// declare the cij scalar
#if GB_IS_PLUS_PAIR_REAL_SEMIRING
    // also initialize cij to zero
    #define GB_CIJ_DECLARE(cij) \
        int64_t cij = 0
#else
    // all other semirings: just declare cij, do not initialize it
    #define GB_CIJ_DECLARE(cij) \
        int64_t cij
#endif

// save the value of C(i,j)
#define GB_CIJ_SAVE(cij,p) Cx [p] = cij

// cij = Cx [pC]
#define GB_GETC(cij,pC) \
    cij = Cx [pC]

// Cx [pC] = cij
#define GB_PUTC(cij,pC) \
    Cx [pC] = cij

// Cx [p] = t
#define GB_CIJ_WRITE(p,t) Cx [p] = t

// C(i,j) += t
#define GB_CIJ_UPDATE(p,t) \
    Cx [p] += t

// x + y
#define GB_ADD_FUNCTION(x,y) \
    x + y

// type with size of GB_CTYPE, and can be used in compare-and-swap
#define GB_CTYPE_PUN \
    int64_t

// bit pattern for bool, 8-bit, 16-bit, and 32-bit integers
#define GB_CTYPE_BITS \
    0

// 1 if monoid update can skipped entirely (the ANY monoid)
#define GB_IS_ANY_MONOID \
    0

// 1 if monoid update is EQ
#define GB_IS_EQ_MONOID \
    0

// 1 if monoid update can be done atomically, 0 otherwise
#define GB_HAS_ATOMIC \
    1

// 1 if monoid update can be done with an OpenMP atomic update, 0 otherwise
#if GB_MICROSOFT
    #define GB_HAS_OMP_ATOMIC \
        1
#else
    #define GB_HAS_OMP_ATOMIC \
        1
#endif

// 1 for the ANY_PAIR semirings
#define GB_IS_ANY_PAIR_SEMIRING \
    0

// 1 if PAIR is the multiply operator 
#define GB_IS_PAIR_MULTIPLIER \
    1

// atomic compare-exchange
#define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
    GB_ATOMIC_COMPARE_EXCHANGE_64 (target, expected, desired)

#if GB_IS_ANY_PAIR_SEMIRING

    // result is purely symbolic; no numeric work to do.  Hx is not used.
    #define GB_HX_WRITE(i,t)
    #define GB_CIJ_GATHER(p,i)
    #define GB_HX_UPDATE(i,t)
    #define GB_CIJ_MEMCPY(p,i,len)

#else

    // Hx [i] = t
    #define GB_HX_WRITE(i,t) Hx [i] = t

    // Cx [p] = Hx [i]
    #define GB_CIJ_GATHER(p,i) Cx [p] = Hx [i]

    // Hx [i] += t
    #define GB_HX_UPDATE(i,t) \
        Hx [i] += t

    // memcpy (&(Cx [p]), &(Hx [i]), len)
    #define GB_CIJ_MEMCPY(p,i,len) \
        memcpy (Cx +(p), Hx +(i), (len) * sizeof(int64_t))

#endif

// disable this semiring and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_PLUS || GxB_NO_PAIR || GxB_NO_INT64 || GxB_NO_PLUS_INT64 || GxB_NO_PAIR_INT64 || GxB_NO_PLUS_PAIR_INT64)

//------------------------------------------------------------------------------
// C=A'*B or C<!M>=A'*B: dot product (phase 2)
//------------------------------------------------------------------------------

GrB_Info GB_Adot2B__plus_pair_int64
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix *Aslice, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *GB_RESTRICT B_slice,
    int64_t *GB_RESTRICT *C_counts,
    int nthreads, int naslice, int nbslice
)
{ 
    // C<M>=A'*B now uses dot3
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #define GB_PHASE_2_OF_2
    #include "GB_AxB_dot2_meta.c"
    #undef GB_PHASE_2_OF_2
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<M>=A'*B: masked dot product method (phase 2)
//------------------------------------------------------------------------------

// #include <assert.h>
#include <x86intrin.h>
// #include <stdio.h>
// #include <stdbool.h>
// #include <stdint.h>
// #include <string.h>
#include <immintrin.h>
// #include <stdlib.h>

// using AVX2
typedef union i4vector64
{
    // in avxintrin.h:
    // typedef long long __v4di __attribute__ ((__vector_size__ (32)));
    __v4di v ;
    int64_t i [4] ;
    __m256i m ;
    __m256d d ;
    double x [4] ;
}
GB_vector ;

void dump (GB_vector x, char *name) ;

void dump (GB_vector x, char *name)
{
    printf ("%s: ", name) ;
    for (int k = 0 ; k < 4 ; k++) printf (" %ld", x.i [k]) ;
    printf ("\n") ;
}

GrB_Info GB_Adot3B__plus_pair_int64
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    const GB_task_struct *GB_RESTRICT TaskList,
    const int ntasks,
    const int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    // #include "GB_AxB_dot3_template.c"
//------------------------------------------------------------------------------
// GB_AxB_dot3_template: C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_DOT3
#define GB_DOT3
#endif

{

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Cp = C->p ;
    const int64_t *GB_RESTRICT Ch = C->h ;
    int64_t  *GB_RESTRICT Ci = C->i ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) (B_is_pattern ? NULL : B->x) ;
    const int64_t bvlen = B->vlen ;
    const int64_t bnvec = B->nvec ;
    const bool B_is_hyper = B->is_hyper ;

    const int64_t *GB_RESTRICT Mi = M->i ;
    const GB_void *GB_RESTRICT Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
    const size_t msize = M->type->size ;

    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const bool A_is_hyper = GB_IS_HYPER (A) ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) (A_is_pattern ? NULL : A->x) ;

    //--------------------------------------------------------------------------
    // C<M> = A'*B
    //--------------------------------------------------------------------------

    // C and M have the same pattern, except some entries of C may become
    // zombies.
    int64_t nzombies = 0 ;

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList [taskid].kfirst ;
        int64_t klast  = TaskList [taskid].klast ;
        int64_t pC_first = TaskList [taskid].pC ;
        int64_t pC_last  = TaskList [taskid].pC_end ;
        int64_t task_nzombies = 0 ;
        int64_t bpleft = 0 ;

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get C(:,k) and M(:k)
            //------------------------------------------------------------------

            int64_t j = (Ch == NULL) ? k : Ch [k] ;
            int64_t pC_start, pC_end ;
            if (k == kfirst)
            { 
                // First vector for task; may only be partially owned.
                pC_start = pC_first ;
                pC_end   = GB_IMIN (Cp [k+1], pC_last) ;
            }
            else if (k == klast)
            { 
                // Last vector for task; may only be partially owned.
                pC_start = Cp [k] ;
                pC_end   = pC_last ;
            }
            else
            { 
                // task fully owns this vector C(:,k).
                pC_start = Cp [k] ;
                pC_end   = Cp [k+1] ;
            }

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            int64_t pB_start, pB_end ;
            GB_lookup (B_is_hyper, Bh, Bp, &bpleft, bnvec-1, j,
                &pB_start, &pB_end) ;
            int64_t bjnz = pB_end - pB_start ;

            //------------------------------------------------------------------
            // C(:,j)<M(:,j)> = A(:,i)'*B(:,j)
            //------------------------------------------------------------------

            if (bjnz == 0)
            {
            
                //--------------------------------------------------------------
                // C(:,j) is empty if B(:,j) is empty
                //--------------------------------------------------------------

                task_nzombies += (pC_end - pC_start) ;
                for (int64_t pC = pC_start ; pC < pC_end ; pC++)
                { 
                    // C(i,j) is a zombie
                    Ci [pC] = GB_FLIP (Mi [pC]) ;
                }
            }
            else
            {

                //--------------------------------------------------------------
                // B(:,j) not empty
                //--------------------------------------------------------------

                int64_t ib_first = Bi [pB_start] ;
                int64_t ib_last  = Bi [pB_end-1] ;
                int64_t apleft = 0 ;

                for (int64_t pC = pC_start ; pC < pC_end ; pC++)
                {

                    //----------------------------------------------------------
                    // compute C(i,j)
                    //----------------------------------------------------------

                    // get the value of M(i,j)
                    int64_t i = Mi [pC] ;
                    if (GB_mcast (Mx, pC, msize))   // note: Mx [pC], same as Cx
                    { 

                        //------------------------------------------------------
                        // M(i,j) is true, so compute C(i,j)
                        //------------------------------------------------------

                        // get A(:,i), if it exists
                        int64_t pA, pA_end ;
                        GB_lookup (A_is_hyper, Ah, Ap, &apleft, anvec-1, i,
                            &pA, &pA_end) ;

                        // C(i,j) = A(:,i)'*B(:,j)
                        // #include "GB_AxB_dot_cij.c"

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

        // v =_mm256_set1_epi64x (int64_t x) ;
        //      loads x into all 4 entries of the vector v:
        //      for (k = 0 to 3) v [k] = x

        // v = _mm256_loadu_si256 ((__m256i const *) x) ;
        //      loads 4 entries from x [0...3] into the vector v:
        //      for (k = 0 to 3) v [k] = x [k]

        // c = _mm256_cmpeq_epi64 (a, b) ;
        //      compares the vectors a and b, for k = 0:3
        //      for (k = 0 to 3) c [k] = (a [k] == b [k]) ? -1 : 0

        // c = _mm256_or_si256 (a,b) ;
        //      logical or:
        //      for (k = 0 to 3) c [k] = a [k] | b [k]

        // mask = _mm256_movemask_pd (c) ;
        //      sets each bit of the mask from most sig. bit of each entry in c
        //      for (k = 0 to 3) mask bit [k] = (most sig. bit of c [k])

        // k = _mm_popcnt_u32 (mask) ;
        //      counts the number of bits set in a 32-bit integer (mask)

        // load the next 4 entries of Ai [pA ...] and broadcast them
        #define GB_BROADCAST(X)                     \
            (X ## 0).m = _mm256_set1_epi64x (X ## i [p ## X +0]) ; \
            (X ## 1).m = _mm256_set1_epi64x (X ## i [p ## X +1]) ; \
            (X ## 2).m = _mm256_set1_epi64x (X ## i [p ## X +2]) ; \
            (X ## 3).m = _mm256_set1_epi64x (X ## i [p ## X +3])

        // load the next entries of Ai [pA ...] and broadcast it
        #define GB_BROADCAST1(X)                        \
            (X ## 0).m = _mm256_set1_epi64x (X ## i [p ## X +0])

        // load the next 4 entries of Bi [pB ...]
        #define GB_LOAD(X) \
            (X ## 0).m = _mm256_loadu_si256 ((__m256i const *)(X ## i + p ## X))

        // munch the next 4 entries of Ai [pA ...] and load the next 4
        #define GB_MUNCH_BROADCAST(X)                   \
        {                                               \
            p ## X += 4 ;                               \
            if (p ## X ## _end - p ## X < 4) break ;    \
            GB_BROADCAST (X) ;                          \
            continue ;                                  \
        }

        // munch the next entries of Ai [pA ...]
        #define GB_MUNCH_BROADCAST1(X)                  \
        {                                               \
            p ## X += 1 ;                               \
            if (p ## X ## _end - p ## X < 1) break ;    \
            GB_BROADCAST1 (X) ;                         \
            continue ;                                  \
        }

        // munch the next 4 entries of Bi [pB ...] and load the next 4
        #define GB_MUNCH_LOAD(X)                        \
        {                                               \
            p ## X += 4 ;                               \
            if (p ## X ## _end - p ## X < 4) break ;    \
            GB_LOAD (X) ;                               \
            continue ;                                  \
        }

#if 0
        if (ainz >= 4 && bjnz >= 4)
        {
            // if (ainz < bjnz)
            {
                GB_vector B0, A0, A1, A2, A3 ;
                GB_BROADCAST (A) ;
                GB_LOAD (B) ;
                while (1)
                {
                    ASSERT (pA_end - pA >= 4) ;
                    ASSERT (pB_end - pB >= 4) ;

                    // skip if last entry of a comes before first entry of b
                    if (A3.i [3] < B0.i [0]) GB_MUNCH_BROADCAST (A) ;

                    // skip if last entry of b comes before first entry of a
                    if (B0.i [3] < A0.i [0]) GB_MUNCH_LOAD (B)  ;

                    // count the intersections

                    // compare (4 instructions)
                    GB_vector C0, C1, C2, C3 ;
                    C0.m = _mm256_cmpeq_epi64 (A0.m, B0.m) ;
                    C1.m = _mm256_cmpeq_epi64 (A1.m, B0.m) ;
                    C2.m = _mm256_cmpeq_epi64 (A2.m, B0.m) ;
                    C3.m = _mm256_cmpeq_epi64 (A3.m, B0.m) ;

                    // or (3 instructions)
                    C0.m = _mm256_or_si256 (C0.m, C1.m) ;
                    C2.m = _mm256_or_si256 (C2.m, C3.m) ;
                    C0.m = _mm256_or_si256 (C0.m, C2.m) ;

                    // count (2 instructions)
                    uint32_t mask = _mm256_movemask_pd (C0.d) ;
                    cij += _mm_popcnt_u32 (mask) ;

                    if (A3.i [3] < B0.i [3])
                    {
                        GB_MUNCH_BROADCAST (A) ;
                    }
                    else
                    {
                        GB_MUNCH_LOAD (B) ;
                    }
                }
            }
            #if 0
            else
            {

                GB_vector A0, B0, B1, B2, B3 ;
                GB_LOAD (A) ;
                GB_BROADCAST (B) ;
                while (1)
                {
                    ASSERT (pA_end - pA >= 4) ;
                    ASSERT (pB_end - pB >= 4) ;

                    // skip if last entry of a comes before first entry of b
                    if (A0.i [3] < B0.i [0]) GB_MUNCH_LOAD (A) ;

                    // skip if last entry of b comes before first entry of a
                    if (B3.i [3] < A0.i [0]) GB_MUNCH_BROADCAST (B)  ;

                    // count the intersections

                    // compare (4 instructions)
                    GB_vector C0, C1, C2, C3 ;
                    C0.m = _mm256_cmpeq_epi64 (A0.m, B0.m) ;
                    C1.m = _mm256_cmpeq_epi64 (A0.m, B1.m) ;
                    C2.m = _mm256_cmpeq_epi64 (A0.m, B2.m) ;
                    C3.m = _mm256_cmpeq_epi64 (A0.m, B3.m) ;

                    // or (3 instructions)
                    C0.m = _mm256_or_si256 (C0.m, C1.m) ;
                    C2.m = _mm256_or_si256 (C2.m, C3.m) ;
                    C0.m = _mm256_or_si256 (C0.m, C2.m) ;

                    // count (2 instructions)
                    uint32_t mask = _mm256_movemask_pd (C0.d) ;
                    cij += _mm_popcnt_u32 (mask) ;

                    if (A0.i [3] < B3.i [3])
                    {
                        GB_MUNCH_LOAD (A) ;
                    }
                    else
                    {
                        GB_MUNCH_BROADCAST (B) ;
                    }
                }
            }
            #endif
        }
#endif

        #if 0
        if (ainz >= 4 && bjnz > 1)
        {
            // 4x1 kernel:  A is long, B is short
            GB_vector A0, B0 ;
            GB_LOAD (A) ;
            GB_BROADCAST1 (B) ;
            while (1)
            {
                ASSERT (pA_end - pA >= 4) ;
                ASSERT (pB_end - pB >= 1) ;

                // skip if last entry of a comes b
                if (A0.i [3] < B0.i [0]) GB_MUNCH_LOAD (A) ;

                // skip if b comes before first entry of a
                if (B0.i [0] < A0.i [0]) GB_MUNCH_BROADCAST1 (B)  ;

                // count the intersections

                // compare (1 instructions)
                GB_vector C0 ;
                C0.m = _mm256_cmpeq_epi64 (A0.m, B0.m) ;

                // count (2 instructions)
                uint32_t mask = _mm256_movemask_pd (C0.d) ;
                cij += _mm_popcnt_u32 (mask) ;

                if (A0.i [3] < B0.i [3])
                {
                    GB_MUNCH_LOAD (A) ;
                }
                else
                {
                    GB_MUNCH_BROADCAST1 (B) ;
                }
            }
        }
        #endif

#if 0
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
#endif
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
#undef GB_MUNCH_BROADCAST
#undef GB_MUNCH_LOAD
#undef GB_BROADCAST
#undef GB_LOAD

                    }
                    else
                    { 

                        //------------------------------------------------------
                        // M(i,j) is false, so C(i,j) is a zombie
                        //------------------------------------------------------

                        task_nzombies++ ;
                        Ci [pC] = GB_FLIP (i) ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // sum up the zombies found by this task
        //----------------------------------------------------------------------

        nzombies += task_nzombies ;
    }

    //--------------------------------------------------------------------------
    // finalize the zombie count for C
    //--------------------------------------------------------------------------

    C->nzombies = nzombies ;
}

#undef GB_DOT3

    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C+=A'*B: dense dot product
//------------------------------------------------------------------------------

GrB_Info GB_Adot4B__plus_pair_int64
(
    GrB_Matrix C,
    const GrB_Matrix A, bool A_is_pattern,
    int64_t *GB_RESTRICT A_slice, int naslice,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *GB_RESTRICT B_slice, int nbslice,
    const int nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_AxB_dot4_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C=A*B, C<M>=A*B, C<!M>=A*B: saxpy3 method (Gustavson + Hash)
//------------------------------------------------------------------------------

#include "GB_AxB_saxpy3_template.h"

GrB_Info GB_Asaxpy3B__plus_pair_int64
(
    GrB_Matrix C,
    const GrB_Matrix M, bool Mask_comp, const bool Mask_struct,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    GB_saxpy3task_struct *GB_RESTRICT TaskList,
    const int ntasks,
    const int nfine,
    const int nthreads,
    GB_Context Context
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_AxB_saxpy3_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

#endif

