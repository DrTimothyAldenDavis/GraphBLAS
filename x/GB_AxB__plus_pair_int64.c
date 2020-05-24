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

// the scalar 1
#define GB_CTYPE_ONE \
    ((int64_t) 1)

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

// declare the cij scalar
#define GB_CIJ_DECLARE(cij) \
    int64_t cij = 0 ;   // TODO, initialize to zero for PLUS_PAIR

#define GB_IS_PLUS_PAIR_REAL_SEMIRING 1

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

{
// printf (" PLUS_PAIR_REAL ") ;
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
// will be C(i,j) = mod (nnz(A(:,j)),2).

// If both A(:,i) and B(:,j) are sparse, then the intersection must still be
// found, so these optimizations can be used only if A(:,i) and/or B(:,j) are
// fully populated.

// For built-in, pre-generated semirings, the PAIR operator is only coupled
// with either the ANY, PLUS, EQ, or XOR monoids, since the other monoids are
// equivalent to the ANY monoid.  With no accumulator, EQ is the same as ANY,
// they differ for the C+=A'*B operation (see *dot4*).

#include "GB_unused.h"

// cij += A(k,i) * B(k,j), for merge operation
#undef  GB_DOT_MERGE

#if GB_IS_PLUS_PAIR_REAL_SEMIRING

    // PLUS_PAIR semiring for integers, float, and double (not complex)
    #define GB_DOT_MERGE cij++ ;

#else

    #define GB_DOT_MERGE                                                \
    {                                                                   \
        GB_GETA (aki, Ax, pA) ;             /* aki = A(k,i) */          \
        GB_GETB (bkj, Bx, pB) ;             /* bkj = B(k,j) */          \
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
    }

#endif

{

    //--------------------------------------------------------------------------
    // get the start of A(:,i) and B(:,j)
    //--------------------------------------------------------------------------

    bool cij_exists = false ;   // C(i,j) not yet in the pattern

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
    // compute C(i,j) = A(:,i)' * B(j,:)
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

        cij_exists = true ;

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
                cij = (GB_CTYPE) bvlen ;
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


    }
    else if (ainz == bvlen)
    {

        //----------------------------------------------------------------------
        // A(:,i) is dense and B(:,j) is sparse
        //----------------------------------------------------------------------

        cij_exists = true ;

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
                cij = (GB_CTYPE) bjnz ;
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

    }
    else if (bjnz == bvlen)
    {

        //----------------------------------------------------------------------
        // A(:,i) is sparse and B(:,j) is dense
        //----------------------------------------------------------------------

        cij_exists = true ;

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
                cij = (GB_CTYPE) ainz ;
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

    }
    else if (ainz > 16 * bjnz)
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
                #if defined ( GB_PHASE_1_OF_2 )
                cij_exists = true ;
                break ;
                #else
                GB_DOT_MERGE
                GB_DOT_TERMINAL (cij) ;         // break if cij == terminal
                pA++ ;
                pB++ ;
                #endif
            }
        }

    }
    else if (bjnz > 16 * ainz)
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
                #if defined ( GB_PHASE_1_OF_2 )
                cij_exists = true ;
                break ;
                #else
                GB_DOT_MERGE
                GB_DOT_TERMINAL (cij) ;         // break if cij == terminal
                pA++ ;
                pB++ ;
                #endif
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A(:,i) and B(:,j) have about the same sparsity
        //----------------------------------------------------------------------

        #if GB_IS_PLUS_PAIR_REAL_SEMIRING

            // PLUS_PAIR semiring
            #if defined ( GB_PHASE_1_OF_2 )

                while (pA < pA_end && pB < pB_end)
                {
                    int64_t ia = Ai [pA] ;
                    int64_t ib = Bi [pB] ;
                    if (ia == ib)
                    {
                        cij_exists = true ;
                        break ;
                    }
                    else
                    {
                        pA += (ia < ib) ;
                        pB += (ib < ia) ;
                    }
                }

            #else

                #define LOAD_A              \
                    a [0] = Ai [pA  ] ;     \
                    a [1] = Ai [pA+1]

                #define LOAD_B              \
                    b [0] = Bi [pB  ] ;     \
                    b [1] = Bi [pB+1]

                #define MUNCH_A             \
                {                           \
                    pA += 2 ;               \
                    if (pA_end - pA < 2)    \
                    {                       \
                        break ;             \
                    }                       \
                    else                    \
                    {                       \
                        LOAD_A ;            \
                        continue ;          \
                    }                       \
                }

                #define MUNCH_B             \
                {                           \
                    pB += 2 ;               \
                    if (pB_end - pB < 2)    \
                    {                       \
                        break ;             \
                    }                       \
                    else                    \
                    {                       \
                        LOAD_B ;            \
                        continue ;          \
                    }                       \
                }

                #define MUNCH_AB            \
                {                           \
                    pA += 2 ;               \
                    pB += 2 ;               \
                    if ((pA_end - pA < 2) || (pB_end - pB < 2))    \
                    {                       \
                        break ;             \
                    }                       \
                    else                    \
                    {                       \
                        LOAD_A ;            \
                        LOAD_B ;            \
                        continue ;          \
                    }                       \
                }

                // cij = the size of the set intersection of Ai [pA .. pA_end]
                // and Bi [pB .. pB_end]
                ASSERT (pA_end - pA == ainz) ;
                ASSERT (pB_end - pB == bjnz) ;
                if (ainz >= 2 && bjnz >= 2)
                {
                    int64_t a [2] ; LOAD_A ;
                    int64_t b [2] ; LOAD_B ;
                    while (1)
                    {
                        // get the next 2 entries from each list
                        ASSERT (pA_end - pA >= 2) ;
                        ASSERT (pB_end - pB >= 2) ;
                        if (a [1] < b [0]) MUNCH_A ;
                        if (b [1] < a [0]) MUNCH_B ;

                        /*
                        cij +=
                            (a [0] == b [0]) + (a [0] == b [1]) +
                            (a [1] == b [0]) + (a [1] == b [1]) ;
                        if (a [1] < b [1]) MUNCH_A
                        else MUNCH_B ;
                        */

                        if (a [1] == b [1])
                        {
                            cij += (a [0] == b [0]) + 1 ;
                            MUNCH_AB ;
                        }

                        cij +=
                            (a [0] == b [0]) + (a [0] == b [1]) +
                            (a [1] == b [0]);//(a [1] == b [1]) ;
                        if (a [1] < b [1]) MUNCH_A else MUNCH_B ;

                    }
                }

                #if 0
                    
                // block 4:

                // TODO use a vectorize load (or typecast to _m256d)
                #define LOAD_A              \
                    a [0] = Ai [pA  ] ;     \
                    a [1] = Ai [pA+1] ;     \
                    a [2] = Ai [pA+2] ;     \
                    a [3] = Ai [pA+3] ;

                #define LOAD_B              \
                    b [0] = Bi [pB  ] ;     \
                    b [1] = Bi [pB+1] ;     \
                    b [2] = Bi [pB+2] ;     \
                    b [3] = Bi [pB+3] ;

                #define MUNCH_A             \
                {                           \
                    pA += 4 ;               \
                    if (pA_end - pA < 4)    \
                    {                       \
                        break ;             \
                    }                       \
                    else                    \
                    {                       \
                        LOAD_A ;            \
                        continue ;          \
                    }                       \
                }

                #define MUNCH_B             \
                {                           \
                    pB += 4 ;               \
                    if (pB_end - pB < 4)    \
                    {                       \
                        break ;             \
                    }                       \
                    else                    \
                    {                       \
                        LOAD_B ;            \
                        continue ;          \
                    }                       \
                }

                // cij = the size of the set intersection of Ai [pA .. pA_end]
                // and Bi [pB .. pB_end]
                ASSERT (pA_end - pA == ainz) ;
                ASSERT (pB_end - pB == bjnz) ;
                if (ainz >= 4 && bjnz >= 4)
                {
                    int64_t a [4] ; LOAD_A ;
                    int64_t b [4] ; LOAD_B ;
                    while (1)
                    {
                        // get the next 4 entries from each list
                        ASSERT (pA_end - pA >= 4) ;
                        ASSERT (pB_end - pB >= 4) ;
                        if (a [3] < b [0]) MUNCH_A ;
                        if (b [3] < a [0]) MUNCH_B ;
                        // TODO: vectorize this statement:
                        cij +=
                            (a [0] == b [0]) + (a [0] == b [1]) + (a [0] == b [2]) + (a [0] == b [3]) +
                            (a [1] == b [0]) + (a [1] == b [1]) + (a [1] == b [2]) + (a [1] == b [3]) +
                            (a [2] == b [0]) + (a [2] == b [1]) + (a [2] == b [2]) + (a [2] == b [3]) +
                            (a [3] == b [0]) + (a [3] == b [1]) + (a [3] == b [2]) + (a [3] == b [3]) ;
                        // TODO: extend this method to all semirings
                        if (a [3] < b [3])
                        { 
                            MUNCH_A ;
                        }
                        else
                        { 
                            MUNCH_B ;
                        }
                    }
                }

                #endif

                // cleanup
                while (pA < pA_end && pB < pB_end)
                {
                    int64_t ia = Ai [pA] ;
                    int64_t ib = Bi [pB] ;
                    if (ia == ib)
                    {
                        cij++ ;
                        pA++ ;
                        pB++ ;
                    }
                    else
                    {
                        pA += (ia < ib) ;
                        pB += (ib < ia) ;
                    }
                }

            #endif

        #else

            while (pA < pA_end && pB < pB_end)
            {
                int64_t ia = Ai [pA] ;
                int64_t ib = Bi [pB] ;
                if (ia == ib)
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    #if defined ( GB_PHASE_1_OF_2 )
                    cij_exists = true ;
                    break ;
                    #else
                    GB_DOT_MERGE
                    GB_DOT_TERMINAL (cij) ;         // break if cij == terminal
                    pA++ ;
                    pB++ ;
                    #endif
                }
                else
                {
                    pA += (ia < ib) ;
                    pB += (ib < ia) ;
                }
            }

        #endif

    }

    //--------------------------------------------------------------------------
    // save C(i,j)
    //--------------------------------------------------------------------------

    #if GB_IS_PLUS_PAIR_REAL_SEMIRING && !defined ( GB_PHASE_1_OF_2 )
    cij_exists = (cij > 0) ;
    #endif

    #if defined ( GB_DOT3 )

        // GB_AxB_dot3: computing C<M>=A'*B
        if (cij_exists)
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
        if (cij_exists)
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

