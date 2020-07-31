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
#include "GB_sort.h"
#include "GB_atomics.h"
#include "GB_AxB_saxpy3.h"
#include "GB_AxB__include.h"
#include "GB_unused.h"

// The C=A*B semiring is defined by the following types and operators:

// A'*B function (dot2):     GB_Adot2B__any_firstj1_int32
// A'*B function (dot3):     GB_Adot3B__any_firstj1_int32
// C+=A'*B function (dot4):  GB_Adot4B__any_firstj1_int32
// A*B function (saxpy3):    GB_Asaxpy3B__any_firstj1_int32

// C type:   int32_t
// A type:   int32_t
// B type:   int32_t

// Multiply: z = (k+1)
// Add:      cij = z
//           'any' monoid?  1
//           atomic?        1
//           OpenMP atomic? 0
// MultAdd:  cij = (k+1)
// Identity: 0
// Terminal: { cij_is_terminal = true ; break ; }

#define GB_ATYPE \
    int32_t

#define GB_BTYPE \
    int32_t

#define GB_CTYPE \
    int32_t

// true for int64, uint64, float, double, float complex, and double complex 
#define GB_CTYPE_IGNORE_OVERFLOW \
    0

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA) \
    ;

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB) \
    ;

#define GB_CX(p) Cx [p]

// multiply operator
#define GB_MULT(z, x, y, i, k, j) \
    z = (k+1)

// cast from a real scalar (or 2, if C is complex) to the type of C
#define GB_CTYPE_CAST(x,y) \
    ((int32_t) x)

// multiply-add
#define GB_MULTADD(z, x, y, i, k, j) \
    z = (k+1)

// monoid identity value
#define GB_IDENTITY \
    0

// break if cij reaches the terminal value (dot product only)
#define GB_DOT_TERMINAL(cij) \
    { cij_is_terminal = true ; break ; }

// simd pragma for dot-product loop vectorization
#define GB_PRAGMA_SIMD_DOT(cij) \
    ;

// simd pragma for other loop vectorization
#define GB_PRAGMA_SIMD_VECTORIZE GB_PRAGMA_SIMD

// 1 for the PLUS_PAIR_(real) semirings, not for the complex case
#define GB_IS_PLUS_PAIR_REAL_SEMIRING \
    0

// declare the cij scalar
#if GB_IS_PLUS_PAIR_REAL_SEMIRING
    // also initialize cij to zero
    #define GB_CIJ_DECLARE(cij) \
        int32_t cij = 0
#else
    // all other semirings: just declare cij, do not initialize it
    #define GB_CIJ_DECLARE(cij) \
        int32_t cij
#endif

// cij = Cx [pC]
#define GB_GETC(cij,p) cij = Cx [p]

// Cx [pC] = cij
#define GB_PUTC(cij,p) Cx [p] = cij

// Cx [p] = t
#define GB_CIJ_WRITE(p,t) Cx [p] = t

// C(i,j) += t
#define GB_CIJ_UPDATE(p,t) \
    Cx [p] = t

// x + y
#define GB_ADD_FUNCTION(x,y) \
    y

// bit pattern for bool, 8-bit, 16-bit, and 32-bit integers
#define GB_CTYPE_BITS \
    0xffffffffL

// 1 if monoid update can skipped entirely (the ANY monoid)
#define GB_IS_ANY_MONOID \
    1

// 1 if monoid update is EQ
#define GB_IS_EQ_MONOID \
    0

// 1 if monoid update can be done atomically, 0 otherwise
#define GB_HAS_ATOMIC \
    1

// 1 if monoid update can be done with an OpenMP atomic update, 0 otherwise
#if GB_MICROSOFT
    #define GB_HAS_OMP_ATOMIC \
        0
#else
    #define GB_HAS_OMP_ATOMIC \
        0
#endif

// 1 for the ANY_PAIR semirings
#define GB_IS_ANY_PAIR_SEMIRING \
    0

// 1 if PAIR is the multiply operator 
#define GB_IS_PAIR_MULTIPLIER \
    0

// 1 if monoid is PLUS_FC32
#define GB_IS_PLUS_FC32_MONOID \
    0

// 1 if monoid is PLUS_FC64
#define GB_IS_PLUS_FC64_MONOID \
    0

// 1 for the FIRSTI or FIRSTI1 multiply operator
#define GB_IS_FIRSTI_MULTIPLIER \
    0

// 1 for the FIRSTJ or FIRSTJ1 multiply operator
#define GB_IS_FIRSTJ_MULTIPLIER \
    1

// 1 for the SECONDJ or SECONDJ1 multiply operator
#define GB_IS_SECONDJ_MULTIPLIER \
    0

// atomic compare-exchange
#define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
    GB_ATOMIC_COMPARE_EXCHANGE_32 (target, expected, desired)

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
        Hx [i] = t

    // memcpy (&(Cx [p]), &(Hx [i]), len)
    #define GB_CIJ_MEMCPY(p,i,len) \
        memcpy (Cx +(p), Hx +(i), (len) * sizeof(int32_t))

#endif

// disable this semiring and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_ANY || GxB_NO_FIRSTJ1 || GxB_NO_INT32 || GxB_NO_ANY_INT32 || GxB_NO_FIRSTJ1_INT32 || GxB_NO_ANY_FIRSTJ1_INT32)

//------------------------------------------------------------------------------
// C=A'*B or C<!M>=A'*B: dot product (phase 2)
//------------------------------------------------------------------------------

GrB_Info GB_Adot2B__any_firstj1_int32
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A, bool A_is_pattern, int64_t *GB_RESTRICT A_slice,
    const GrB_Matrix B, bool B_is_pattern, int64_t *GB_RESTRICT B_slice,
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

GrB_Info GB_Adot3B__any_firstj1_int32
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
    #include "GB_AxB_dot3_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C+=A'*B: dense dot product
//------------------------------------------------------------------------------

GrB_Info GB_Adot4B__any_firstj1_int32
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

GrB_Info GB_Asaxpy3B__any_firstj1_int32
(
    GrB_Matrix C,
    const GrB_Matrix M, bool Mask_comp, const bool Mask_struct,
    const bool M_dense_in_place,
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
    // #include "GB_AxB_saxpy3_template.c"
//------------------------------------------------------------------------------
// GB_AxB_saxpy3_template: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_AxB_saxpy3_template.c computes C=A*B for any semiring and matrix types.
// It is #include'd in GB_AxB_saxpy3 to construct the generic method (for
// arbitary user-defined operators and/or typecasting), and in the hard-coded
// GB_Asaxpy3B* workers in the Generated/ folder.

#include "GB_unused.h"

//------------------------------------------------------------------------------
// template code for C=A*B via the saxpy3 method
//------------------------------------------------------------------------------

{

double ttt = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // get the chunk size
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Cp = C->p ;                // ok: C is sparse
    // const int64_t *GB_RESTRICT Ch = C->h ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnvec = C->nvec ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) (B_is_pattern ? NULL : B->x) ;
    const int64_t bvlen = B->vlen ;
    // const int64_t bnvec = B->nvec ;
    // const bool B_is_hyper = (Bh != NULL) ;
    const bool B_jumbled = B->jumbled ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const bool A_is_hyper = GB_IS_HYPER (A) ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) (A_is_pattern ? NULL : A->x) ;
    const bool A_jumbled = A->jumbled ;

    const int64_t *GB_RESTRICT Mp = NULL ;
    const int64_t *GB_RESTRICT Mh = NULL ;
    const int64_t *GB_RESTRICT Mi = NULL ;
    const GB_void *GB_RESTRICT Mx = NULL ;
    size_t msize = 0 ;
    int64_t mnvec = 0 ;
    int64_t mvlen = 0 ;
    bool M_is_hyper = false ;
    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mi = M->i ;
        Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
        mnvec = M->nvec ;
        mvlen = M->vlen ;
        M_is_hyper = (Mh != NULL) ;
    }

    // 3 cases:
    //      M not present and Mask_comp false: compute C=A*B
    //      M present     and Mask_comp false: compute C<M>=A*B
    //      M present     and Mask_comp true : compute C<!M>=A*B
    // If M is NULL on input, then Mask_comp is also false on input.

    bool mask_is_M = (M != NULL && !Mask_comp) ;

    //==========================================================================
    // phase2: numeric work for fine tasks
    //==========================================================================

    // Coarse tasks: nothing to do in phase2.
    // Fine tasks: compute nnz (C(:,j)), and values in Hx via atomics.

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kk = TaskList [taskid].vector ;
        int team_size = TaskList [taskid].team_size ;
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        int64_t pB     = TaskList [taskid].start ;
        int64_t pB_end = TaskList [taskid].end + 1 ;
        int64_t pleft = 0, pright = anvec-1 ;
        int64_t j = GBH (Bh, kk) ;

        #if GB_IS_SECONDJ_MULTIPLIER
        // SECONDJ or SECONDJ1 multiplier
        // t = aik*bkj = j or j+1
        GB_CIJ_DECLARE (t) ;
        GB_MULT (t, ignore, ignore, i, k, j) ;
        #endif

        #if !GB_IS_ANY_PAIR_SEMIRING
        GB_CTYPE *GB_RESTRICT Hx = (GB_CTYPE *) TaskList [taskid].Hx ;
        #endif

        #if GB_IS_PLUS_FC32_MONOID
        float  *GB_RESTRICT Hx_real = (float *) Hx ;
        float  *GB_RESTRICT Hx_imag = Hx_real + 1 ;
        #elif GB_IS_PLUS_FC64_MONOID
        double *GB_RESTRICT Hx_real = (double *) Hx ;
        double *GB_RESTRICT Hx_imag = Hx_real + 1 ;
        #endif

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // phase2: fine Gustavson task
            //------------------------------------------------------------------

            // Hf [i] == 0: unlocked, i has not been seen in C(:,j).
            //      Hx [i] is not initialized.
            //      M(i,j) is 0, or M is not present.
            //      if M: Hf [i] stays equal to 0 (or 3 if locked)
            //      if !M, or no M: C(i,j) is a new entry seen for 1st time

            // Hf [i] == 1: unlocked, i has not been seen in C(:,j).
            //      Hx [i] is not initialized.  M is present.
            //      M(i,j) is 1. (either M or !M case)
            //      if M: C(i,j) is a new entry seen for the first time.
            //      if !M: Hf [i] stays equal to 1 (or 3 if locked)

            // Hf [i] == 2: unlocked, i has been seen in C(:,j).
            //      Hx [i] is initialized.  This case is independent of M.

            // Hf [i] == 3: locked.  Hx [i] cannot be accessed.

            int8_t *GB_RESTRICT Hf = (int8_t *GB_RESTRICT) TaskList [taskid].Hf;

            if (M == NULL)
            {

                //--------------------------------------------------------------
                // phase2: fine Gustavson task, C(:,j)=A*B(:,j)
                //--------------------------------------------------------------

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

            }
            else if (mask_is_M)
            {

                //--------------------------------------------------------------
                // phase2: fine Gustavson task, C(:,j)<M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                // Hf [i] is 0 if M(i,j) not present or M(i,j)=0.
                // 0 -> 1 : has already been done in phase0 if M(i,j)=1.
                // If M(:,j) is dense, then it is not scattered into Hf.

                // 0 -> 0 : to ignore, if M(i,j)=0
                // 1 -> 3 : to lock, if i seen for first time
                // 2 -> 3 : to lock, if i seen already
                // 3 -> 2 : to unlock; now i has been seen

                GB_GET_M_j ;                // get M(:,j)
                GB_GET_M_j_RANGE (16) ;     // get first and last in M(:,j)
                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                { 
                    int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
                    GB_GET_A_k ;                // get A(:,k)
                    if (aknz == 0) continue ;
                    GB_GET_B_kj ;               // bkj = B(k,j)

                    #if GB_IS_ANY_MONOID

                    #define GB_IKJ                                             \
                        int8_t f ;                                             \
                        GB_ATOMIC_READ                                         \
                        f = Hf [i] ;            /* grab the entry */           \
                        if (f == 0 || f == 2) continue ;                       \
                        GB_ATOMIC_WRITE                                        \
                        Hf [i] = 2 ;            /* unlock the entry */         \
                        GB_MULT_A_ik_B_kj ;     /* t = A(i,k) * B(k,j) */      \
                        GB_ATOMIC_WRITE_HX (i, t) ;    /* Hx [i] = t */

                    #else

                    #define GB_IKJ                                             \
                    {                                                          \
                        GB_MULT_A_ik_B_kj ;     /* t = A(i,k) * B(k,j) */      \
                        int8_t f ;                                             \
                        GB_ATOMIC_READ                                         \
                        f = Hf [i] ;            /* grab the entry */           \
                        if (GB_HAS_ATOMIC && (f == 2))                         \
                        {                                                      \
                            /* C(i,j) already seen; update it */               \
                            GB_ATOMIC_UPDATE_HX (i, t) ; /* Hx [i] += t */     \
                            continue ;       /* C(i,j) has been updated */     \
                        }                                                      \
                        if (f == 0) continue ; /* M(i,j)=0; ignore C(i,j)*/    \
                        do  /* lock the entry */                               \
                        {                                                      \
                            /* do this atomically: */                          \
                            /* { f = Hf [i] ; Hf [i] = 3 ; } */                \
                            GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;            \
                        } while (f == 3) ; /* lock owner gets f=1 or 2 */      \
                        if (f == 1)                                            \
                        {                                                      \
                            /* C(i,j) is a new entry */                        \
                            GB_ATOMIC_WRITE_HX (i, t) ; /* Hx [i] = t */       \
                        }                                                      \
                        else /* f == 2 */                                      \
                        {                                                      \
                            /* C(i,j) already appears in C(:,j) */             \
                            GB_ATOMIC_UPDATE_HX (i, t) ; /* Hx [i] += t */     \
                        }                                                      \
                        GB_ATOMIC_WRITE                                        \
                        Hf [i] = 2 ;                /* unlock the entry */     \
                    }
                    #endif

                    GB_SCAN_M_j_OR_A_k ;
                    #undef GB_IKJ
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase2: fine Gustavson task, C(:,j)<!M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                // Hf [i] is 0 if M(i,j) not present or M(i,j)=0.
                // 0 -> 1 : has already been done in phase0 if M(i,j)=1

                // If M(:,j) is dense, then it is not scattered into Hf.

                // 1 -> 1 : to ignore, if M(i,j)=1
                // 0 -> 3 : to lock, if i seen for first time
                // 2 -> 3 : to lock, if i seen already
                // 3 -> 2 : to unlock; now i has been seen

                GB_GET_M_j ;                // get M(:,j)
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
                        GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                        int8_t f ;

                        #if GB_IS_ANY_MONOID

                        GB_ATOMIC_READ
                        f = Hf [i] ;            // grab the entry
                        if (f == 1 || f == 2) continue ;
                        GB_ATOMIC_WRITE
                        Hf [i] = 2 ;                // unlock the entry
                        GB_ATOMIC_WRITE_HX (i, t) ;    // Hx [i] = t

                        #else

                        GB_ATOMIC_READ
                        f = Hf [i] ;            // grab the entry
                        #if GB_HAS_ATOMIC
                        if (f == 2)             // if true, update C(i,j)
                        {
                            GB_ATOMIC_UPDATE_HX (i, t) ;   // Hx [i] += t
                            continue ;          // C(i,j) has been updated
                        }
                        #endif
                        if (f == 1) continue ; // M(i,j)=1; ignore C(i,j)
                        do  // lock the entry
                        {
                            // do this atomically:
                            // { f = Hf [i] ; Hf [i] = 3 ; }
                            GB_ATOMIC_CAPTURE_INT8 (f, Hf [i], 3) ;
                        } while (f == 3) ; // lock owner of gets f=0 or 2
                        if (f == 0)
                        { 
                            // C(i,j) is a new entry
                            GB_ATOMIC_WRITE_HX (i, t) ;    // Hx [i] = t
                        }
                        else // f == 2
                        { 
                            // C(i,j) already seen
                            GB_ATOMIC_UPDATE_HX (i, t) ;   // Hx [i] += t
                        }
                        GB_ATOMIC_WRITE
                        Hf [i] = 2 ;                // unlock the entry
                        #endif
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // phase2: fine hash task
            //------------------------------------------------------------------

            // Each hash entry Hf [hash] splits into two parts, (h,f).  f
            // is in the 2 least significant bits.  h is 62 bits, and is
            // the 1-based index i of the C(i,j) entry stored at that
            // location in the hash table.

            // If M is present (M or !M), and M(i,j)=1, then (i+1,1)
            // has been inserted into the hash table, in phase0.

            // Given Hf [hash] split into (h,f)

            // h == 0, f == 0: unlocked and unoccupied.
            //                  note that if f=0, h must be zero too.

            // h == i+1, f == 1: unlocked, occupied by M(i,j)=1.
            //                  C(i,j) has not been seen, or is ignored.
            //                  Hx is not initialized.  M is present.
            //                  if !M: this entry will be ignored in C.

            // h == i+1, f == 2: unlocked, occupied by C(i,j).
            //                  Hx is initialized.  M is no longer
            //                  relevant.

            // h == (anything), f == 3: locked.

            int64_t *GB_RESTRICT
                Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
            int64_t hash_bits = (hash_size-1) ;

            if (M == NULL)
            {

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)=A*B(:,j)
                //--------------------------------------------------------------

                // no mask present
                #undef GB_CHECK_MASK_ij
                #include "GB_AxB_saxpy3_fineHash_phase2.c"

            }
            else if (mask_is_M)
            {

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)<M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                GB_GET_M_j ;                // get M(:,j)
                if (M_dense_in_place)
                { 
                    // M(:,j) is dense.  M is not scattered into Hf.
                    if (Mx == NULL)
                    {
                        // Full structural mask, not complemented.
                        // The Mask is ignored, and C(:,j)=A*B(:,j)
                        // TODO: remove this case in caller
                        #include "GB_AxB_saxpy3_fineHash_phase2.c"
                    }
                    #undef  GB_CHECK_MASK_ij
                    #define GB_CHECK_MASK_ij if (Mask [i] == 0) continue ;
                    switch (msize)
                    {
                        default:
                        case 1:
                        {
                            #define M_TYPE uint8_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 2:
                        {
                            #define M_TYPE uint16_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 4:
                        {
                            #define M_TYPE uint32_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 8:
                        {
                            #define M_TYPE uint64_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 16:
                        {
                            #define M_TYPE uint64_t
                            #define M_SIZE 2
                            #undef  GB_CHECK_MASK_ij
                            #define GB_CHECK_MASK_ij                        \
                                if (Mask [2*i] == 0 && Mask [2*i+1] == 0)   \
                                    continue ;
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                    }
                }

                // Given Hf [hash] split into (h,f)

                // h == 0  , f == 0 : unlocked, unoccupied. C(i,j) ignored
                // h == i+1, f == 1 : unlocked, occupied by M(i,j)=1.
                //                    C(i,j) has not been seen.
                //                    Hx is not initialized.
                // h == i+1, f == 2 : unlocked, occupied by C(i,j), M(i,j)=1
                //                    Hx is initialized.
                // h == ..., f == 3 : locked.

                // 0 -> 0 : to ignore, if M(i,j)=0
                // 1 -> 3 : to lock, if i seen for first time
                // 2 -> 3 : to lock, if i seen already
                // 3 -> 2 : to unlock; now i has been seen

                GB_GET_M_j_RANGE (16) ;     // get first and last in M(:,j)
                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                { 
                    int64_t k = GBI (Bi, pB, bvlen) ;       // get B(k,j)
                    GB_GET_A_k ;                // get A(:,k)
                    if (aknz == 0) continue ;
                    GB_GET_B_kj ;               // bkj = B(k,j)
                    #define GB_IKJ                                             \
                    {                                                          \
                        GB_MULT_A_ik_B_kj ;      /* t = A(i,k) * B(k,j) */     \
                        int64_t i1 = i + 1 ;     /* i1 = one-based index */    \
                        int64_t i_unlocked = (i1 << 2) + 2 ;  /* (i+1,2) */    \
                        for (GB_HASH (i))        /* find i in hash table */    \
                        {                                                      \
                            int64_t hf ;                                       \
                            GB_ATOMIC_READ                                     \
                            hf = Hf [hash] ;        /* grab the entry */       \
                            if (GB_HAS_ATOMIC && (hf == i_unlocked))           \
                            {                                                  \
                                /* Hx [hash] += t */                           \
                                GB_ATOMIC_UPDATE_HX (hash, t) ;                \
                                break ;     /* C(i,j) has been updated */      \
                            }                                                  \
                            if (hf == 0) break ; /* M(i,j)=0; ignore Cij */    \
                            if ((hf >> 2) == i1) /* if true, i found */        \
                            {                                                  \
                                do /* lock the entry */                        \
                                {                                              \
                                    /* do this atomically: */                  \
                                    /* { hf = Hf [hash] ; Hf [hash] |= 3 ; }*/ \
                                    GB_ATOMIC_CAPTURE_INT64_OR (hf,Hf[hash],3);\
                                } while ((hf & 3) == 3) ; /* own: f=1,2 */     \
                                if ((hf & 3) == 1) /* f == 1 */                \
                                {                                              \
                                    /* C(i,j) is a new entry in C(:,j) */      \
                                    /* Hx [hash] = t */                        \
                                    GB_ATOMIC_WRITE_HX (hash, t) ;             \
                                }                                              \
                                else /* f == 2 */                              \
                                {                                              \
                                    /* C(i,j) already appears in C(:,j) */     \
                                    /* Hx [hash] += t */                       \
                                    GB_ATOMIC_UPDATE_HX (hash, t) ;            \
                                }                                              \
                                GB_ATOMIC_WRITE                                \
                                Hf [hash] = i_unlocked ; /* unlock entry */    \
                                break ;                                        \
                            }                                                  \
                        }                                                      \
                    }
                    GB_SCAN_M_j_OR_A_k ;
                    #undef GB_IKJ
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase2: fine hash task, C(:,j)<!M(:,j)>=A*B(:,j)
                //--------------------------------------------------------------

                GB_GET_M_j ;                // get M(:,j)
                if (M_dense_in_place)
                { 
                    // M(:,j) is dense.  M is not scattered into Hf.
                    if (Mx == NULL)
                    {
                        // structural mask, complemented.  No work to do.
                        // TODO: remove this case in caller
                        continue ;
                    }
                    #undef  GB_CHECK_MASK_ij
                    #define GB_CHECK_MASK_ij if (Mask [i] != 0) continue ;
                    switch (msize)
                    {
                        default:
                        case 1:
                        {
                            #define M_TYPE uint8_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 2:
                        {
                            #define M_TYPE uint16_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 4:
                        {
                            #define M_TYPE uint32_t
                            // #include "GB_AxB_saxpy3_fineHash_phase2.c"

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_fineHash_phase2_template:
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // phase2: fine hash task, C(:,j)=A*B(:,j)
    //--------------------------------------------------------------------------

    // Given Hf [hash] split into (h,f)

    // h == 0  , f == 0 : unlocked and unoccupied.
    // h == i+1, f == 2 : unlocked, occupied by C(i,j).
    //                    Hx is initialized.
    // h == ..., f == 3 : locked.

    // 0 -> 3 : to lock, if i seen for first time
    // 2 -> 3 : to lock, if i seen already
    // 3 -> 2 : to unlock; now i has been seen

    #ifdef GB_CHECK_MASK_ij
    #ifndef M_SIZE
    #define M_SIZE 1
    #endif
    const M_TYPE *GB_RESTRICT Mask = ((M_TYPE *) Mx) + (M_SIZE * pM_start) ;
    #endif

    if (team_size == 1)
    {
printf ("HERE I AM one\n") ;
        //----------------------------------------------------------------------
        // single-threaded version
        //----------------------------------------------------------------------

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
                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j)
                // is protected by the mask
                GB_CHECK_MASK_ij ;
                #endif
                GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                int64_t i1 = i + 1 ;        // i1 = one-based index
                int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)

//#define GB_HASH_FUNCTION(i) ((((i) << 8) + (i)) & (hash_bits))
//#define GB_HASH(i) int64_t hash = GB_HASH_FUNCTION (i) ; ; GB_REHASH (hash,i)
// #define GB_REHASH(hash,i) hash = ((hash + 1) & (hash_bits))

int64_t hash = GB_HASH_FUNCTION (i) ;

{
int64_t hf = Hf [hash] ;    // grab the entry
if (hf == i_unlocked)       // if true, update C(i,j)
{ 
    // hash entry occuppied by C(i,j): update it
    GB_HX_UPDATE (hash, t) ;    // Hx [hash] += t
    continue ;         // C(i,j) has been updated
}
if (hf == 0)
{ 
    // hash entry unoccuppied: fill it with C(i,j)
    // Hx [hash] = t
    GB_HX_WRITE (hash, t) ;
    Hf [hash] = i_unlocked ; // unlock entry
    continue ;
}
// otherwise: hash table occupied, but not with i
}

                //for (GB_HASH (i))           // find i in hash table

                while (1)
                {
                    GB_REHASH (hash, i) ;
                    // hash++ ;
                    // hash &= hash_bits ;

                    int64_t hf = Hf [hash] ;    // grab the entry
                    if (hf == i_unlocked)       // if true, update C(i,j)
                    { 
                        // hash entry occuppied by C(i,j): update it
                        GB_HX_UPDATE (hash, t) ;    // Hx [hash] += t
                        break ;         // C(i,j) has been updated
                    }
                    if (hf == 0)
                    { 
                        // hash entry unoccuppied: fill it with C(i,j)
                        // Hx [hash] = t
                        GB_HX_WRITE (hash, t) ;
                        Hf [hash] = i_unlocked ; // unlock entry
                        break ;
                    }
                    // otherwise: hash table occupied, but not with i
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // multi-threaded version
        //----------------------------------------------------------------------

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
                #ifdef GB_CHECK_MASK_ij
                // check mask condition and skip if C(i,j)
                // is protected by the mask
                GB_CHECK_MASK_ij ;
                #endif
                GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                int64_t i1 = i + 1 ;        // i1 = one-based index
                int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)
                for (GB_HASH (i))           // find i in hash table
                {
                    int64_t hf ;
                    GB_ATOMIC_READ
                    hf = Hf [hash] ;        // grab the entry
                    #if GB_HAS_ATOMIC
                    if (hf == i_unlocked)  // if true, update C(i,j)
                    {
                        GB_ATOMIC_UPDATE_HX (hash, t) ;// Hx [.]+=t
                        break ;         // C(i,j) has been updated
                    }
                    #endif
                    int64_t h = (hf >> 2) ;
                    if (h == 0 || h == i1)
                    {
                        // h=0: unoccupied, h=i1: occupied by i
                        do  // lock the entry
                        {
                            // do this atomically:
                            // { hf = Hf [hash] ; Hf [hash] |= 3 ; }
                            GB_ATOMIC_CAPTURE_INT64_OR (hf,Hf[hash],3) ;
                        } while ((hf & 3) == 3) ; // owner: f=0 or 2
                        if (hf == 0) // f == 0
                        { 
                            // C(i,j) is a new entry in C(:,j)
                            // Hx [hash] = t
                            GB_ATOMIC_WRITE_HX (hash, t) ;
                            GB_ATOMIC_WRITE
                            Hf [hash] = i_unlocked ; // unlock entry
                            break ;
                        }
                        if (hf == i_unlocked) // f == 2
                        { 
                            // C(i,j) already appears in C(:,j)
                            // Hx [hash] += t
                            GB_ATOMIC_UPDATE_HX (hash, t) ;
                            GB_ATOMIC_WRITE
                            Hf [hash] = i_unlocked ; // unlock entry
                            break ;
                        }
                        // hash table occupied, but not with i
                        GB_ATOMIC_WRITE
                        Hf [hash] = hf ;  // unlock with prior value
                    }
                }
            }
        }
    }

    continue ;
}

#undef M_TYPE
#undef M_SIZE

                        }
                        case 8:
                        {
                            #define M_TYPE uint64_t
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                        case 16:
                        {
                            #define M_TYPE uint64_t
                            #define M_SIZE 2
                            #undef  GB_CHECK_MASK_ij
                            #define GB_CHECK_MASK_ij                        \
                                if (Mask [2*i] != 0 || Mask [2*i+1] != 0)   \
                                    continue ;
                            #include "GB_AxB_saxpy3_fineHash_phase2.c"
                        }
                    }
                }

                // Given Hf [hash] split into (h,f)

                // h == 0  , f == 0 : unlocked and unoccupied.
                // h == i+1, f == 1 : unlocked, occupied by M(i,j)=1.
                //                    C(i,j) is ignored.
                // h == i+1, f == 2 : unlocked, occupied by C(i,j).
                //                    Hx is initialized.

                // h == (anything), f == 3: locked.

                // 1 -> 1 : to ignore, if M(i,j)=1
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
                        GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                        int64_t i1 = i + 1 ;        // i1 = one-based index
                        int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)
                        int64_t i_masked   = (i1 << 2) + 1 ;    // (i+1,1)
                        for (GB_HASH (i))           // find i in hash table
                        {
                            int64_t hf ;
                            GB_ATOMIC_READ
                            hf = Hf [hash] ;        // grab the entry
                            #if GB_HAS_ATOMIC
                            if (hf == i_unlocked)  // if true, update C(i,j)
                            {
                                GB_ATOMIC_UPDATE_HX (hash, t) ;// Hx [.]+=t
                                break ;         // C(i,j) has been updated
                            }
                            #endif
                            if (hf == i_masked) break ; // M(i,j)=1; ignore
                            int64_t h = (hf >> 2) ;
                            if (h == 0 || h == i1)
                            {
                                // h=0: unoccupied, h=i1: occupied by i
                                do // lock the entry
                                {
                                    // do this atomically:
                                    // { hf = Hf [hash] ; Hf [hash] |= 3 ; }
                                    GB_ATOMIC_CAPTURE_INT64_OR (hf,Hf[hash],3) ;
                                } while ((hf & 3) == 3) ; // owner: f=0,1,2
                                if (hf == 0)            // f == 0
                                { 
                                    // C(i,j) is a new entry in C(:,j)
                                    // Hx [hash] = t
                                    GB_ATOMIC_WRITE_HX (hash, t) ;
                                    GB_ATOMIC_WRITE
                                    Hf [hash] = i_unlocked ; // unlock entry
                                    break ;
                                }
                                if (hf == i_unlocked)   // f == 2
                                { 
                                    // C(i,j) already appears in C(:,j)
                                    // Hx [hash] += t
                                    GB_ATOMIC_UPDATE_HX (hash, t) ;
                                    GB_ATOMIC_WRITE
                                    Hf [hash] = i_unlocked ; // unlock entry
                                    break ;
                                }
                                // hash table occupied, but not with i,
                                // or with i but M(i,j)=1 so C(i,j) ignored
                                GB_ATOMIC_WRITE
                                Hf [hash] = hf ;  // unlock with prior value
                            }
                        }
                    }
                }
            }
        }
    }

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (9, ttt) ;
ttt = omp_get_wtime ( ) ;

    //==========================================================================
    // phase3/phase4: count nnz(C(:,j)) for fine tasks, cumsum of Cp
    //==========================================================================

    int64_t cjnz_max = GB_AxB_saxpy3_cumsum (C, TaskList,
        nfine, chunk, nthreads) ;

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (10, ttt) ;
ttt = omp_get_wtime ( ) ;

    //==========================================================================
    // phase5: numeric phase for coarse tasks, gather for fine tasks
    //==========================================================================

    // allocate Ci and Cx
    int64_t cnz = Cp [cnvec] ;      // ok: C is sparse
    GrB_Info info = GB_ix_alloc (C, cnz, true, true, Context) ; // ok: C sparse
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    int64_t  *GB_RESTRICT Ci = C->i ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;

    #if GB_IS_ANY_PAIR_SEMIRING

        // TODO: create C as a constant-value matrix.

        // ANY_PAIR semiring: result is purely symbolic
        int64_t pC ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (pC = 0 ; pC < cnz ; pC++)
        { 
            Cx [pC] = GB_CTYPE_CAST (1, 0) ;
        }

        // Just a precaution; these variables are not used below.  Any attempt
        // to access them will lead to a compile error.
        #define Cx is not used
        #define Hx is not used

        // these have been renamed to ANY_PAIR:
        // EQ_PAIR
        // LAND_PAIR
        // LOR_PAIR
        // MAX_PAIR
        // MIN_PAIR
        // TIMES_PAIR

    #endif

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (11, ttt) ;
ttt = omp_get_wtime ( ) ;

    bool C_jumbled = false ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(||:C_jumbled)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        #if !GB_IS_ANY_PAIR_SEMIRING
        GB_CTYPE *GB_RESTRICT Hx = (GB_CTYPE *) TaskList [taskid].Hx ;
        #endif
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;

        if (taskid < nfine)
        {

            //------------------------------------------------------------------
            // fine task: gather pattern and values
            //------------------------------------------------------------------

            int64_t kk = TaskList [taskid].vector ;
            int team_size = TaskList [taskid].team_size ;
            int leader    = TaskList [taskid].leader ;
            int my_teamid = taskid - leader ;
            int64_t pC = Cp [kk] ;      // ok: C is sparse

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase5: fine Gustavson task, C=A*B, C<M>=A*B, or C<!M>=A*B
                //--------------------------------------------------------------

                // Hf [i] == 2 if C(i,j) is an entry in C(:,j)
                int8_t *GB_RESTRICT
                    Hf = (int8_t *GB_RESTRICT) TaskList [taskid].Hf ;
                int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cvlen, my_teamid, team_size) ;
                if (cjnz == cvlen)
                {
                    // TODO: if all of C is dense, skip this step and
                    // free the pattern of C.
                    // C(:,j) is dense
                    for (int64_t i = istart ; i < iend ; i++)
                    { 
                        Ci [pC + i] = i ;           // ok: C is sparse
                    }
                    #if !GB_IS_ANY_PAIR_SEMIRING
                    // copy Hx [istart:iend-1] into Cx [pC+istart:pC+iend-1]
                    GB_CIJ_MEMCPY (pC + istart, istart, iend - istart) ;

                    // TODO: if C is a single vector, skip the memcpy of
                    // Hx into Cx.  Instead, free C->x and transplant
                    // C->x = Hx, and do not free Hx.
                    #endif
                }
                else
                {
                    // C(:,j) is sparse
                    pC += TaskList [taskid].my_cjnz ;
                    for (int64_t i = istart ; i < iend ; i++)
                    {
                        if (Hf [i] == 2)
                        { 
                            GB_CIJ_GATHER (pC, i) ; // Cx [pC] = Hx [i]
                            Ci [pC++] = i ;         // ok: C is sparse
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase5: fine hash task, C=A*B, C<M>=A*B, C<!M>=A*B
                //--------------------------------------------------------------

                // (Hf [hash] & 3) == 2 if C(i,j) is an entry in C(:,j),
                // and the index i of the entry is (Hf [hash] >> 2) - 1.

                int64_t *GB_RESTRICT
                    Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
                int64_t mystart, myend ;
                GB_PARTITION (mystart, myend, hash_size, my_teamid, team_size) ;
                pC += TaskList [taskid].my_cjnz ;
                for (int64_t hash = mystart ; hash < myend ; hash++)
                {
                    int64_t hf = Hf [hash] ;
                    if ((hf & 3) == 2)
                    { 
                        int64_t i = (hf >> 2) - 1 ; // found C(i,j) in hash
                        Ci [pC] = i ;               // ok: C is sparse
                        // added after deleting phase 6:
                        GB_CIJ_GATHER (pC, hash) ;  // Cx [pC] = Hx [hash]
                        pC++ ;
                    }
                }
                C_jumbled = true ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // numeric coarse task: compute C(:,kfirst:klast)
            //------------------------------------------------------------------

            int64_t *GB_RESTRICT
                Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
            int64_t kfirst = TaskList [taskid].start ;
            int64_t klast = TaskList [taskid].end ;
            int64_t nk = klast - kfirst + 1 ;
            int64_t mark = 2*nk + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase5: coarse Gustavson task
                //--------------------------------------------------------------

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase5: coarse Gustavson task, C=A*B
                    //----------------------------------------------------------

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_B_j ;                // get B(:,j)
                        mark++ ;
                        if (cjnz == cvlen)          // C(:,j) is dense
                        { 
                            GB_COMPUTE_DENSE_C_j ;  // C(:,j) = A*B(:,j)
                        }
                        else if (bjnz == 1)         // C(:,j) = A(:,k)*B(k,j)
                        { 
                            GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE ;
                        }
                        else if (16 * cjnz > cvlen) // C(:,j) is not very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                // scan A(:,k)
                                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                                {
                                    // get A(i,k)
                                    int64_t i = GBI (Ai, pA, avlen) ;
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    if (Hf [i] != mark)
                                    { 
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark ;
                                        GB_HX_WRITE (i, t) ;    // Hx [i] = t
                                    }
                                    else
                                    { 
                                        // C(i,j) += A(i,k) * B(k,j)
                                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t
                                    }
                                }
                            }
                            GB_GATHER_ALL_C_j(mark) ;   // gather into C(:,j) 
                        }
                        else    // C(:,j) is very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                // scan A(:,k)
                                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                                {
                                    // get A(i,k)
                                    int64_t i = GBI (Ai, pA, avlen) ;
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    if (Hf [i] != mark)
                                    { 
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark ;
                                        GB_HX_WRITE (i, t) ; // Hx [i] = t
                                        Ci [pC++] = i ;      // ok: C is sparse
                                    }
                                    else
                                    { 
                                        // C(i,j) += A(i,k) * B(k,j)
                                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t
                                    }
                                }
                            }
                            GB_SORT_AND_GATHER_C_j ;    // gather into C(:,j)
                        }
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase5: coarse Gustavson task, C<M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.

                    // Hf [i] < mark    : M(i,j)=0, C(i,j) is ignored.
                    // Hf [i] == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // Hf [i] == mark+1 : M(i,j)=1, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_B_j ;                // get B(:,j)
                        if (cjnz == cvlen)          // C(:,j) is dense
                        { 
                            GB_COMPUTE_DENSE_C_j ;  // C(:,j) = A*B(:,j)
                            continue ;              // no need to examine M(:,j)
                        }
                        GB_GET_M_j ;            // get M(:,j)
                        GB_GET_M_j_RANGE (64) ; // get first and last in M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        // scatter M(:,j)
                        GB_SCATTER_M_j (pM_start, pM_end, mark) ;
                        if (16 * cjnz > cvlen)  // C(:,j) is not very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            { 
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                #define GB_IKJ                                 \
                                {                                              \
                                    int64_t hf = Hf [i] ;                      \
                                    if (hf == mark)                            \
                                    {                                          \
                                        /* C(i,j) = A(i,k) * B(k,j) */         \
                                        Hf [i] = mark1 ;     /* mark as seen */\
                                        GB_MULT_A_ik_B_kj ;  /* t = aik*bkj */ \
                                        GB_HX_WRITE (i, t) ; /* Hx [i] = t */  \
                                    }                                          \
                                    else if (hf == mark1)                      \
                                    {                                          \
                                        /* C(i,j) += A(i,k) * B(k,j) */        \
                                        GB_MULT_A_ik_B_kj ;  /* t = aik*bkj */ \
                                        GB_HX_UPDATE (i, t) ;/* Hx [i] += t */ \
                                    }                                          \
                                }
                                GB_SCAN_M_j_OR_A_k ;
                                #undef GB_IKJ
                            }
                            GB_GATHER_ALL_C_j(mark1) ;  // gather into C(:,j) 
                        }
                        else    // C(:,j) is very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            { 
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                #define GB_IKJ                                 \
                                {                                              \
                                    int64_t hf = Hf [i] ;                      \
                                    if (hf == mark)                            \
                                    {                                          \
                                        /* C(i,j) = A(i,k) * B(k,j) */         \
                                        Hf [i] = mark1 ;     /* mark as seen */\
                                        GB_MULT_A_ik_B_kj ;  /* t = aik*bkj */ \
                                        GB_HX_WRITE (i, t) ; /* Hx [i] = t */  \
                                        Ci [pC++] = i ; /* C(:,j) pattern */   \
                                    }                                          \
                                    else if (hf == mark1)                      \
                                    {                                          \
                                        /* C(i,j) += A(i,k) * B(k,j) */        \
                                        GB_MULT_A_ik_B_kj ;  /* t = aik*bkj */ \
                                        GB_HX_UPDATE (i, t) ;/* Hx [i] += t */ \
                                    }                                          \
                                }
                                GB_SCAN_M_j_OR_A_k ;
                                #undef GB_IKJ
                            }
                            GB_SORT_AND_GATHER_C_j ;    // gather into C(:,j)
                        }
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase5: coarse Gustavson task, C<!M>=A*B
                    //----------------------------------------------------------

                    // Since the mask is !M:
                    // Hf [i] < mark    : M(i,j)=0, C(i,j) is not yet seen.
                    // Hf [i] == mark   : M(i,j)=1, so C(i,j) is ignored.
                    // Hf [i] == mark+1 : M(i,j)=0, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_B_j ;                // get B(:,j)
                        if (cjnz == cvlen)          // C(:,j) is dense
                        { 
                            GB_COMPUTE_DENSE_C_j ;  // C(:,j) = A*B(:,j)
                            continue ;              // no need to examine M(:,j)
                        }
                        GB_GET_M_j ;            // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        // scatter M(:,j)
                        GB_SCATTER_M_j (pM_start, pM_end, mark) ;
                        if (16 * cjnz > cvlen)  // C(:,j) is not very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                // scan A(:,k)
                                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                                {
                                    // get A(i,k)
                                    int64_t i = GBI (Ai, pA, avlen) ;
                                    int64_t hf = Hf [i] ;
                                    if (hf < mark)
                                    { 
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark1 ;     // mark as seen
                                        GB_MULT_A_ik_B_kj ;  // t =A(i,k)*B(k,j)
                                        GB_HX_WRITE (i, t) ; // Hx [i] = t
                                    }
                                    else if (hf == mark1)
                                    { 
                                        // C(i,j) += A(i,k) * B(k,j)
                                        GB_MULT_A_ik_B_kj ;  // t =A(i,k)*B(k,j)
                                        GB_HX_UPDATE (i, t) ;// Hx [i] += t
                                    }
                                }
                            }
                            GB_GATHER_ALL_C_j(mark1) ;  // gather into C(:,j) 
                        }
                        else    // C(:,j) is very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                // scan A(:,k)
                                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                                {
                                    // get A(i,k)
                                    int64_t i = GBI (Ai, pA, avlen) ;
                                    int64_t hf = Hf [i] ;
                                    if (hf < mark)
                                    { 
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark1 ;        // mark as seen
                                        GB_MULT_A_ik_B_kj ;  // t =A(i,k)*B(k,j)
                                        GB_HX_WRITE (i, t) ;    // Hx [i] = t
                                        Ci [pC++] = i ; // create C(:,j) pattern
                                    }
                                    else if (hf == mark1)
                                    { 
                                        // C(i,j) += A(i,k) * B(k,j)
                                        GB_MULT_A_ik_B_kj ;  // t =A(i,k)*B(k,j)
                                        GB_HX_UPDATE (i, t) ;   // Hx [i] += t
                                    }
                                }
                            }
                            GB_SORT_AND_GATHER_C_j ;    // gather into C(:,j)
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase5: coarse hash task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;
                int64_t hash_bits = (hash_size-1) ;

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C=A*B
                    //----------------------------------------------------------

                    // no mask present
                    #undef GB_CHECK_MASK_ij
                    // printf ("coarse hash phase 5 no mask\n") ;
                    #include "GB_AxB_saxpy3_coarseHash_phase5.c"

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    if (M_dense_in_place)
                    { 
                        // M(:,j) is dense.  M is not scattered into Hf.
                        if (Mx == NULL)
                        {
                            // Full structural mask, not complemented.
                            // The Mask is ignored, and C(:,j)=A*B(:,j)
                            // TODO: remove this case in caller
                            // printf ("coarse hash phase 5 M mask struct\n") ;
                            #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                        }
                        #define GB_CHECK_MASK_ij if (Mask [i] == 0) continue ;
                        switch (msize)
                        {
                            default:
                            case 1:
                            {
                                #define M_TYPE uint8_t
                                // printf ("coarse hash phase 5 M 1\n") ;
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 2:
                            {
                                // printf ("coarse hash phase 5 M 2\n") ;
                                #define M_TYPE uint16_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 4:
                            {
                                // printf ("coarse hash phase 5 M 3\n") ;
                                #define M_TYPE uint32_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 8:
                            {
                                // printf ("coarse hash phase 5 M 8\n") ;
                                #define M_TYPE uint64_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 16:
                            {
                                // printf ("coarse hash phase 5 M 16\n") ;
                                #define M_TYPE uint64_t
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                      \
                                    if (Mask [2*i] == 0 && Mask [2*i+1] == 0) \
                                        continue ;
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                        }
                    }

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark            : M(i,j)=0, C(i,j) is ignored.
                    // h == i, f == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // h == i, f == mark+1 : M(i,j)=1, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_M_j ;                // get M(:,j)
                        GB_GET_M_j_RANGE (64) ;     // get 1st & last in M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;               // hash M(:,j)
                        GB_GET_B_j ;                // get B(:,j)
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        { 
                            int64_t k = GBI (Bi, pB, bvlen) ;  // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
                            GB_GET_B_kj ;               // bkj = B(k,j)
                            #define GB_IKJ                                     \
                            {                                                  \
                                for (GB_HASH (i))       /* find i in hash */   \
                                {                                              \
                                    int64_t f = Hf [hash] ;                    \
                                    if (f < mark) break ; /* M(i,j)=0, ignore*/\
                                    if (Hi [hash] == i)                        \
                                    {                                          \
                                        GB_MULT_A_ik_B_kj ; /* t = aik*bkj */  \
                                        if (f == mark) /* if true, i is new */ \
                                        {                                      \
                                            /* C(i,j) is new */                \
                                            Hf [hash] = mark1 ; /* mark seen */\
                                            GB_HX_WRITE (hash, t) ;/*Hx[.]=t */\
                                            Ci [pC++] = i ;                    \
                                        }                                      \
                                        else                                   \
                                        {                                      \
                                            /* C(i,j) has been seen; update */ \
                                            GB_HX_UPDATE (hash, t) ;           \
                                        }                                      \
                                        break ;                                \
                                    }                                          \
                                }                                              \
                            }
                            GB_SCAN_M_j_OR_A_k ;
                            #undef GB_IKJ
                        }
                        // found i if: Hf [hash] == mark1 and Hi [hash] == i
                        GB_SORT_AND_GATHER_HASHED_C_j (mark1, Hi [hash] == i) ;
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase5: coarse hash task, C<!M>=A*B
                    //----------------------------------------------------------

                    if (M_dense_in_place)
                    { 
                        // M(:,j) is dense.  M is not scattered into Hf.
                        if (Mx == NULL)
                        {
                            // structural mask, complemented.  No work to do.
                            // TODO: remove this case in caller
                            continue ;
                        }
                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij if (Mask [i] != 0) continue ;
                        switch (msize)
                        {
                            default:
                            case 1:
                            {
                                // printf ("coarse hash phase 5 !M 1\n") ;
                                #define M_TYPE uint8_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 2:
                            {
                                // printf ("coarse hash phase 5 !M 2\n") ;
                                #define M_TYPE uint16_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 4:
                            {
                                // printf ("coarse hash phase 5 !M 4\n") ;
                                #define M_TYPE uint32_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 8:
                            {
                                // printf ("coarse hash phase 5 !M 8\n") ;
                                #define M_TYPE uint64_t
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                            case 16:
                            {
                                // printf ("coarse hash phase 5 !M 16\n") ;
                                #define M_TYPE uint64_t
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                      \
                                    if (Mask [2*i] != 0 || Mask [2*i+1] != 0) \
                                        continue ;
                                #include "GB_AxB_saxpy3_coarseHash_phase5.c"
                            }
                        }
                    }

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark: unoccupied, M(i,j)=0, and C(i,j) not yet seen.
                    // h == i, f == mark   : M(i,j)=1. C(i,j) ignored.
                    // h == i, f == mark+1 : M(i,j)=0, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;      // ok: C is sparse
                        int64_t cjnz = Cp [kk+1] - pC ;     // ok: C is sparse
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_M_j ;                // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;               // hash M(:,j)
                        GB_GET_B_j ;                // get B(:,j)
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
                                for (GB_HASH (i))       // find i in hash
                                {
                                    int64_t f = Hf [hash] ;
                                    if (f < mark)   // if true, i is new
                                    { 
                                        // C(i,j) is new
                                        Hf [hash] = mark1 ; // mark C(i,j) seen
                                        Hi [hash] = i ;
                                        GB_MULT_A_ik_B_kj ; // t = A(i,k)*B(k,j)
                                        GB_HX_WRITE (hash, t) ; // Hx [hash] = t
                                        Ci [pC++] = i ;         // ok: C sparse
                                        break ;
                                    }
                                    if (Hi [hash] == i)
                                    {
                                        if (f == mark1)
                                        { 
                                            // C(i,j) has been seen; update it.
                                            GB_MULT_A_ik_B_kj ;//t=A(i,k)*B(k,j)
                                            GB_HX_UPDATE (hash, t) ;//Hx[ ] += t
                                        }
                                        break ;
                                    }
                                }
                            }
                        }
                        // found i if: Hf [hash] == mark1 and Hi [hash] == i
                        GB_SORT_AND_GATHER_HASHED_C_j (mark1, Hi [hash] == i) ;
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // log the state of C->jumbled
    //--------------------------------------------------------------------------

    C->jumbled = C_jumbled ;

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (12, ttt) ;

}

#undef Cx
#undef Hx

    return (GrB_SUCCESS) ;
    #endif
}

#endif

