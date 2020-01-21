
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
#include "GB_saxpy3.h"
#include "GB_AxB__include.h"

// The C=A*B semiring is defined by the following types and operators:

// A'*B function (dot2):     GB_Adot2B__any_second_int32
// A'*B function (dot3):     GB_Adot3B__any_second_int32
// C+=A'*B function (dot4):  GB_Adot4B__any_second_int32
// A*B function (saxpy3):    GB_Asaxpy3B__any_second_int32

// C type:   int32_t
// A type:   int32_t
// B type:   int32_t

// Multiply: z = bkj
// Add:      cij = z
//           any monoid?    1
//           atomic?        1
//           OpenMP atomic? 0
// MultAdd:  cij = bkj
// Identity: 0
// Terminal: break ;

#define GB_ATYPE \
    int32_t

#define GB_BTYPE \
    int32_t

#define GB_CTYPE \
    int32_t

// aik = Ax [pA]
#define GB_GETA(aik,Ax,pA) \
    ;

// bkj = Bx [pB]
#define GB_GETB(bkj,Bx,pB) \
    int32_t bkj = Bx [pB]

#define GB_CX(p) Cx [p]

// multiply operator
#define GB_MULT(z, x, y) \
    z = y

// multiply-add
#define GB_MULTADD(z, x, y) \
    z = y

// monoid identity value
#define GB_IDENTITY \
    0

// break if cij reaches the terminal value (dot product only)
#define GB_DOT_TERMINAL(cij) \
    break ;

// simd pragma for dot product
#define GB_DOT_SIMD \
    ;

// declare the cij scalar
#define GB_CIJ_DECLARE(cij) \
    int32_t cij

// save the value of C(i,j)
#define GB_CIJ_SAVE(cij,p) Cx [p] = cij

// cij = Cx [pC]
#define GB_GETC(cij,pC) \
    cij = Cx [pC]

// Cx [pC] = cij
#define GB_PUTC(cij,pC) \
    Cx [pC] = cij

// For saxpy3:

// Cx [p] = t
#define GB_CIJ_WRITE(p,t) Cx [p] = t

// C(i,j) += t
#define GB_CIJ_UPDATE(p,t) \
    Cx [p] = t

// x + y
#define GB_ADD_FUNCTION(x,y) \
    y

// type with size of GB_CTYPE, and can be used in compare-and-swap
#define GB_CTYPE_PUN \
    int32_t

// 1 if monoid update can skipped entirely (the ANY monoid)
#define GB_IS_ANY_MONOID \
    1

// 1 if monoid update can be done atomically, 0 otherwise
#define GB_HAS_ATOMIC \
    1

// 1 if monoid update can be done with a #pragma omp atomic update, 0 otherwise
#define GB_HAS_OMP_ATOMIC \
    0

// 1 for the ANY_PAIR semirings
#define GB_IS_ANY_PAIR_SEMIRING \
    0

// 1 if PAIR is the multiply operator 
#define GB_IS_PAIR_MULTIPLIER \
    0

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
    (GxB_NO_ANY || GxB_NO_SECOND || GxB_NO_INT32 || GxB_NO_ANY_INT32 || GxB_NO_SECOND_INT32 || GxB_NO_ANY_SECOND_INT32)

//------------------------------------------------------------------------------
// C=A'*B or C<!M>=A'*B: dot product (phase 2)
//------------------------------------------------------------------------------

GrB_Info GB_Adot2B__any_second_int32
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

GrB_Info GB_Adot3B__any_second_int32
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

GrB_Info GB_Adot4B__any_second_int32
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

GrB_Info GB_Asaxpy3B__any_second_int32
(
    GrB_Matrix *Chandle,
    const GrB_Matrix M, bool Mask_comp, const bool Mask_struct,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    GB_saxpy3task_struct *GB_RESTRICT *TaskList_handle,
    void *Work [3], size_t Worksize [3],
    const int ntasks,
    const int nfine,
    const int nthreads,
    GB_Context Context
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    // get copies of these pointers for GB_FREE_ALL
    GB_saxpy3task_struct *GB_RESTRICT TaskList = (*TaskList_handle) ;
    GrB_Matrix C = (*Chandle) ;
    int64_t *Hi_all = Work [0] ;
    int64_t *Hf_all = Work [1] ;
    GB_void *Hx_all = Work [2] ;
    size_t Hi_size_total = Worksize [0] ;
    size_t Hf_size_total = Worksize [1] ;
    size_t Hx_size_total = Worksize [2] ;
    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    // #include "GB_AxB_saxpy3_template.c"
{

//------------------------------------------------------------------------------
// GB_AxB_saxpy3_template: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_AxB_saxpy3_template.c computes C=A*B for any semiring and matrix types.
// It is #include'd in GB_AxB_saxpy3 to construct the generic method (for
// arbitary run-time defined operators and/or typecasting), in the hard-coded
// GB_Asaxpy3B* workers in the Generated/ folder, and in the
// compile-time-constructed functions called by GB_AxB_user.

//------------------------------------------------------------------------------
// macros (TODO: put these in GB_AxB_saxpy3.h)
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GB_GET_M_j: prepare to iterate over M(:,j)
//------------------------------------------------------------------------------

// prepare to iterate over the vector M(:,j), for the (kk)th vector of B
// TODO lookup all M(:,j) for all vectors in B, in a single pass,
// and save the mapping (like C_to_M mapping in GB_ewise_slice)
#define GB_GET_M_j                                              \
    int64_t mpleft = 0 ;                                        \
    int64_t mpright = mnvec-1 ;                                 \
    int64_t pM, pM_end ;                                        \
    GB_lookup (M_is_hyper, Mh, Mp, &mpleft, mpright,            \
        ((Bh == NULL) ? kk : Bh [kk]), &pM, &pM_end) ;          \
    int64_t mjnz = pM_end - pM ;    /* nnz (M (:,j)) */

//------------------------------------------------------------------------------
// GB_GET_M_j_RANGE: get the first and last indices in M(:,j)
//------------------------------------------------------------------------------

#define GB_GET_M_j_RANGE(gamma)                                 \
    int64_t im_first = -1, im_last = -1 ;                       \
    if (mjnz > 0)                                               \
    {                                                           \
        im_first = Mi [pM] ;        /* get first M(:,j) */      \
        im_last  = Mi [pM_end-1] ;  /* get last M(:,j) */       \
    }                                                           \
    int64_t mjnz_much = mjnz * gamma

//------------------------------------------------------------------------------
// GB_SCATTER_M_j: scatter M(:,j) for a coarse Gustavson task
//------------------------------------------------------------------------------

// scatter M(:,j) for a coarse Gustavson task, C<M>=A*B or C<!M>=A*B
#define GB_SCATTER_M_j                                          \
    for (int64_t p = pM ; p < pM_end ; p++) /* scan M(:,j) */   \
    {                                                           \
        GB_GET_M_ij (p) ;               /* get M(i,j) */        \
        if (mij) Hf [Mi [p]] = mark ;   /* Hf [i] = M(i,j) */   \
    }

//------------------------------------------------------------------------------
// GB_HASH_M_j: scatter M(:,j) for a coarse hash task
//------------------------------------------------------------------------------

// hash M(:,j) into Hf and Hi for coarse hash task, C<M>=A*B or C<!M>=A*B
#define GB_HASH_M_j                                             \
    for (int64_t p = pM ; p < pM_end ; p++) /* scan M(:,j) */   \
    {                                                           \
        GB_GET_M_ij (p) ;       /* get M(i,j) */                \
        if (!mij) continue ;    /* skip if M(i,j)=0 */          \
        int64_t i = Mi [p] ;                                    \
        for (GB_HASH (i))       /* find i in hash */            \
        {                                                       \
            if (Hf [hash] < mark)                               \
            {                                                   \
                Hf [hash] = mark ;  /* insert M(i,j)=1 */       \
                Hi [hash] = i ;                                 \
                break ;                                         \
            }                                                   \
        }                                                       \
    }

//------------------------------------------------------------------------------
// GB_HASH1_M_j: scatter M(:,j) for a 1-vector coarse hash task
//------------------------------------------------------------------------------

// hash M(:,j) for a 1-vector coarse hash task, C<M>=A*B or C<!M>=A*B
#define GB_HASH1_M_j                                            \
    for (int64_t p = pM ; p < pM_end ; p++) /* scan M(:,j) */   \
    {                                                           \
        GB_GET_M_ij (p) ;      /* get M(i,j) */                 \
        if (!mij) continue ;    /* skip if M(i,j)=0 */          \
        int64_t i = Mi [p] ;                                    \
        int64_t i_mark = ((i+1) << 2) + 1 ; /* (i+1,1) */       \
        for (GB_HASH (i))       /* find i in hash table */      \
        {                                                       \
            if (Hf [hash] == 0) /* if true, entry unoccupied */ \
            {                                                   \
                Hf [hash] = i_mark ;    /* M(i,j) = 1 */        \
                break ;                                         \
            }                                                   \
        }                                                       \
    }

//------------------------------------------------------------------------------
// GB_GET_B_j: prepare to iterate over B(:,j)
//------------------------------------------------------------------------------

// prepare to iterate over the vector B(:,j), the (kk)th vector in B,
// where j == ((Bh == NULL) ? kk : Bh [kk]).  Note that j itself is never
// needed; just kk.
#define GB_GET_B_j                                                          \
    int64_t pleft = 0 ;                                                     \
    int64_t pright = anvec-1 ;                                              \
    int64_t pB = Bp [kk] ;                                                  \
    int64_t pB_end = Bp [kk+1] ;                                            \
    int64_t bjnz = pB_end - pB ;  /* nnz (B (:,j) */                        \
    /* TODO can skip if mjnz == 0 for C<M>=A*B tasks */                     \
    if (A_is_hyper && bjnz > 2)                                             \
    {                                                                       \
        /* trim Ah [0..pright] to remove any entries past last B(:,j), */   \
        /* to speed up GB_lookup in GB_GET_A_k. */                          \
        GB_bracket_right (Bi [pB_end-1], Ah, 0, &pright) ;                  \
    }

//------------------------------------------------------------------------------
// GB_GET_B_kj: get the numeric value of B(k,j)
//------------------------------------------------------------------------------

#define GB_GET_B_kj \
    GB_GETB (bkj, Bx, pB)       /* bkj = Bx [pB] */

//------------------------------------------------------------------------------
// GB_GET_A_k: prepare to iterate over the vector A(:,k)
//------------------------------------------------------------------------------

#define GB_GET_A_k                                                      \
    int64_t pA, pA_end ;                                                \
    GB_lookup (A_is_hyper, Ah, Ap, &pleft, pright, k, &pA, &pA_end) ;   \
    int64_t aknz = pA_end - pA ;    /* nnz (A (:,k)) */

//------------------------------------------------------------------------------
// GB_SKIP_IF_A_k_DISJOINT_WITH_M_j:  skip if A(:,k) and M(:,j) are disjoint
//------------------------------------------------------------------------------

// skip C(:,j)<M> += A(:,k)*B(k,j) if A(:,k) and M(:,j), for C<M>=A*B methods
#define GB_SKIP_IF_A_k_DISJOINT_WITH_M_j                    \
    if (aknz == 0) continue ;                               \
    int64_t alo = Ai [pA] ;     /* get first A(:,k) */      \
    int64_t ahi = Ai [pA_end-1] ;   /* get last A(:,k) */   \
    if (ahi < im_first || alo > im_last) continue

//------------------------------------------------------------------------------
// GB_GET_M_ij: get the numeric value of M(i,j)
//------------------------------------------------------------------------------

#define GB_GET_M_ij(pM)                             \
    /* get M(i,j), at Mi [pM] and Mx [pM] */        \
    bool mij = GB_mcast (Mx, pM, msize)

//------------------------------------------------------------------------------
// GB_MULT_A_ik_B_kj: declare t and compute t = A(i,k) * B(k,j)
//------------------------------------------------------------------------------

#if GB_IS_PAIR_MULTIPLIER

    // PAIR multiplier: t is always 1; no numeric work to do to compute t.
    // The LXOR_PAIR and PLUS_PAIR semirings need the value t = 1 to use in
    // their monoid operator, however.
    #define t 1
    #define GB_MULT_A_ik_B_kj

#else

    // typical semiring
    #define GB_MULT_A_ik_B_kj                                   \
        GB_GETA (aik, Ax, pA) ;     /* aik = Ax [pA] ;  */      \
        GB_CIJ_DECLARE (t) ;        /* ctype t ;        */      \
        GB_MULT (t, aik, bkj)       /* t = aik * bkj ;  */

#endif

//------------------------------------------------------------------------------
// GB_COMPUTE_DENSE_C_j: compute C(:,j)=A*B(:,j) when C(:,j) is completely dense
//------------------------------------------------------------------------------

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_COMPUTE_DENSE_C_j                                \
        for (int64_t i = 0 ; i < cvlen ; i++)                   \
        {                                                       \
            Ci [pC + i] = i ;                                   \
        }

#else

    // typical semiring
    #define GB_COMPUTE_DENSE_C_j                                \
        for (int64_t i = 0 ; i < cvlen ; i++)                   \
        {                                                       \
            Ci [pC + i] = i ;                                   \
            GB_CIJ_WRITE (pC + i, GB_IDENTITY) ; /* C(i,j)=0 */ \
        }                                                       \
        for ( ; pB < pB_end ; pB++)     /* scan B(:,j) */       \
        {                                                       \
            int64_t k = Bi [pB] ;       /* get B(k,j) */        \
            GB_GET_A_k ;                /* get A(:,k) */        \
            if (aknz == 0) continue ;                           \
            GB_GET_B_kj ;               /* bkj = B(k,j) */      \
            /* TODO handle the case when A(:,k) is dense */     \
            for ( ; pA < pA_end ; pA++) /* scan A(:,k) */       \
            {                                                   \
                int64_t i = Ai [pA] ;   /* get A(i,k) */        \
                GB_MULT_A_ik_B_kj ;     /* t = A(i,k)*B(k,j) */ \
                GB_CIJ_UPDATE (pC + i, t) ; /* Cx [pC+i]+=t */  \
            }                                                   \
        }

#endif

//------------------------------------------------------------------------------
// GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE: compute C(:,j) when nnz(B(:,j)) == 1
//------------------------------------------------------------------------------

// C(:,j) = A(:,k)*B(k,j) when there is a single entry in B(:,j)
#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE                      \
        int64_t k = Bi [pB] ;       /* get B(k,j) */                \
        GB_GET_A_k ;                /* get A(:,k) */                \
        memcpy (Ci + pC, Ai + pA, aknz * sizeof (int64_t)) ;

#else

    // typical semiring
    #define GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE                      \
        int64_t k = Bi [pB] ;       /* get B(k,j) */                \
        GB_GET_A_k ;                /* get A(:,k) */                \
        GB_GET_B_kj ;               /* bkj = B(k,j) */              \
        for ( ; pA < pA_end ; pA++) /* scan A(:,k) */               \
        {                                                           \
            int64_t i = Ai [pA] ;       /* get A(i,k) */            \
            GB_MULT_A_ik_B_kj ;         /* t = A(i,k)*B(k,j) */     \
            GB_CIJ_WRITE (pC, t) ;      /* Cx [pC] = t */           \
            Ci [pC++] = i ;                                         \
        }

#endif

//------------------------------------------------------------------------------
// GB_GATHER_ALL_C_j: gather the values and pattern of C(:,j)
//------------------------------------------------------------------------------

// gather the pattern and values of C(:,j) for a coarse Gustavson task (no sort)
#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_GATHER_ALL_C_j(mark)                                 \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            if (Hf [i] == mark)                                     \
            {                                                       \
                Ci [pC++] = i ;                                     \
            }                                                       \
        }

#else

    // typical semiring
    #define GB_GATHER_ALL_C_j(mark)                                 \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            if (Hf [i] == mark)                                     \
            {                                                       \
                GB_CIJ_GATHER (pC, i) ; /* Cx [pC] = Hx [i] */      \
                Ci [pC++] = i ;                                     \
            }                                                       \
        }

#endif

//------------------------------------------------------------------------------
// GB_SORT_AND_GATHER_C_j: sort the pattern of C(:,j) and gather values
//------------------------------------------------------------------------------

// sort the pattern of C(:,j) then gather the values for a coarse Gustavson task
#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; just sort the pattern
    #define GB_SORT_AND_GATHER_C_j                              \
        /* sort the pattern of C(:,j) */                        \
        GB_qsort_1a (Ci + Cp [kk], cjnz) ;

#else

    // typical semiring
    #define GB_SORT_AND_GATHER_C_j                              \
        /* sort the pattern of C(:,j) */                        \
        GB_qsort_1a (Ci + Cp [kk], cjnz) ;                      \
        /* gather the values into C(:,j) */                     \
        for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)      \
        {                                                       \
            int64_t i = Ci [pC] ;                               \
            GB_CIJ_GATHER (pC, i) ;   /* Cx [pC] = Hx [i] */    \
        }

#endif

//------------------------------------------------------------------------------
// GB_SORT_AND_GATHER_HASHED_C_j: sort pattern, gather values, for coarse hash 
//------------------------------------------------------------------------------

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; just sort the pattern
    #define GB_SORT_AND_GATHER_HASHED_C_j(hash_mark,Hi_hash_equals_i)       \
        /* sort the pattern of C(:,j) */                                    \
        GB_qsort_1a (Ci + Cp [kk], cjnz) ;

#else

    // sort the pattern of C(:,j) then gather the values for a coarse hash task
    #define GB_SORT_AND_GATHER_HASHED_C_j(hash_mark,Hi_hash_equals_i)       \
        /* sort the pattern of C(:,j) */                                    \
        GB_qsort_1a (Ci + Cp [kk], cjnz) ;                                  \
        /* gather the values of C(:,j) from Hf for 1-vector coarse hash */  \
        /* tasks or both Hf and Hi for multi-vector coarse hash tasks */    \
        for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)                  \
        {                                                                   \
            int64_t i = Ci [pC] ;                                           \
            int64_t marked = (hash_mark) ;                                  \
            for (GB_HASH (i))           /* find i in hash table */          \
            {                                                               \
                if (Hf [hash] == marked && (Hi_hash_equals_i))              \
                {                                                           \
                    /* i found in the hash table */                         \
                    /* Cx [pC] = Hx [hash] ; */                             \
                    GB_CIJ_GATHER (pC, hash) ;                              \
                    break ;                                                 \
                }                                                           \
            }                                                               \
        }

#endif

//------------------------------------------------------------------------------
// GB_SCAN_M_j_OR_A_k: compute C(:,j) using linear scan or binary search
//------------------------------------------------------------------------------

// C(:,j)<M(:,j)>=A(:,k)*B(k,j) using one of two methods
#define GB_SCAN_M_j_OR_A_k                                              \
{                                                                       \
    if (aknz > 256 && mjnz_much < aknz)                                 \
    /* nnz(M(:,j)) much less than nnz(A(:,k)) */                        \
    {                                                                   \
        /* scan M(:,j), and do binary search for A(i,k) */              \
        for (int64_t p = pM ; p < pM_end ; p++)                         \
        {                                                               \
            int64_t i = Mi [p] ;    /* get M(i,j) */                    \
            GB_GET_M_ij (p) ;                                           \
            if (!mij) continue ;    /* skip if M(i,j)=0 */              \
            bool found ;            /* search for A(i,k) */             \
            int64_t apright = pA_end - 1 ;                              \
            GB_BINARY_SEARCH (i, Ai, pA, apright, found) ;              \
            if (found)                                                  \
            {                                                           \
                /* C(i,j)<M(i,j)> += A(i,k) * B(k,j) for this method. */\
                /* M(i,j) is now known to be equal to 1, so there are */\
                /* cases in the GB_IKJ operation that can never */      \
                /* occur.  This could be pruned from the GB_IKJ */      \
                /* operation, but then this operation would differ */   \
                /* from the GB_IKJ operation in the linear-time scan */ \
                /* of A(:,j), below.  It's unlikely that pruning this */\
                /* case would lead to much performance improvement. */  \
                GB_IKJ ;                                                \
            }                                                           \
        }                                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        /* scan A(:,k), and lookup M(i,j) */                            \
        for ( ; pA < pA_end ; pA++)                                     \
        {                                                               \
            int64_t i = Ai [pA] ;   /* get A(i,k) */                    \
            /* do C(i,j)<M(i,j)> += A(i,k) * B(k,j) for this method */  \
            /* M(i,j) may be 0 or 1, as given in the hash table */      \
            GB_IKJ ;                                                    \
        }                                                               \
    }                                                                   \
}

//------------------------------------------------------------------------------
// GB_ATOMIC_UPDATE:  Hx [i] += t
//------------------------------------------------------------------------------

#if GB_IS_ANY_MONOID

    // The update Hx [i] += t can be skipped entirely, for the ANY monoid.
    #define GB_ATOMIC_UPDATE(i,t)

#elif GB_HAS_ATOMIC

    // Hx [i] += t via atomic update
    #if GB_HAS_OMP_ATOMIC
        // built-in PLUS, TIMES, LOR, LAND, LXOR monoids can be
        // implemented with an OpenMP pragma
        #define GB_ATOMIC_UPDATE(i,t)       \
            GB_PRAGMA (omp atomic update)   \
            GB_HX_UPDATE (i, t)
    #else
        // built-in MIN, MAX, and EQ monoids only, which cannot
        // be implemented with an OpenMP pragma
        #define GB_ATOMIC_UPDATE(i,t)                               \
            GB_CTYPE xold, xnew, *px = Hx + (i) ;                   \
            do                                                      \
            {                                                       \
                /* xold = Hx [i] via atomic read */                 \
                GB_PRAGMA (omp atomic read)                         \
                xold = (*px) ;                                      \
                /* xnew = xold + t */                               \
                xnew = GB_ADD_FUNCTION (xold, t) ;                  \
            }                                                       \
            /* TODO use __atomic_*, not __sync_* methods */         \
            while (!__sync_bool_compare_and_swap                    \
                ((GB_CTYPE_PUN *) px,                               \
                * ((GB_CTYPE_PUN *) (&xold)),                       \
                * ((GB_CTYPE_PUN *) (&xnew))))
    #endif

#else

    // Hx [i] += t can only be done inside the critical section
    #define GB_ATOMIC_UPDATE(i,t)       \
        GB_PRAGMA (omp flush)           \
        GB_HX_UPDATE (i, t) ;           \
        GB_PRAGMA (omp flush)

#endif

//------------------------------------------------------------------------------
// GB_ATOMIC_WRITE:  Hx [i] = t
//------------------------------------------------------------------------------

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_ATOMIC_WRITE(i,t)

#else 

    // atomic write
    #if GB_HAS_ATOMIC
        // Hx [i] = t via atomic write
        #define GB_ATOMIC_WRITE(i,t)       \
            GB_PRAGMA (omp atomic write)   \
            GB_HX_WRITE (i, t)
    #else
        // Hx [i] = t via critical section
        #define GB_ATOMIC_WRITE(i,t)       \
            GB_PRAGMA (omp flush)          \
            GB_HX_WRITE (i, t) ;           \
            GB_PRAGMA (omp flush)
    #endif

#endif

//------------------------------------------------------------------------------
// hash
//------------------------------------------------------------------------------

// to iterate over the hash table, looking for index i:
// for (GB_HASH (i)) { ... }
#define GB_HASH(i) int64_t hash = GB_HASH_FUNCTION (i) ; ; GB_REHASH (hash,i)

//------------------------------------------------------------------------------
// free workspace
//------------------------------------------------------------------------------

// See also GB_FREE_ALL in GB_AxB_saxpy3.  It is #define'd here for the
// hard-coded GB_Asaxpy3B* functions, and for the user-defined functions called
// by GB_AxB_user.
#ifndef GB_FREE_ALL
#define GB_FREE_ALL                                                         \
{                                                                           \
    GB_FREE_MEMORY (*(TaskList_handle), ntasks, sizeof (GB_saxpy3task_struct));\
    GB_FREE_MEMORY (Hi_all, Hi_size_total, sizeof (int64_t)) ;              \
    GB_FREE_MEMORY (Hf_all, Hf_size_total, sizeof (int64_t)) ;              \
    GB_FREE_MEMORY (Hx_all, Hx_size_total, 1) ;                             \
    GB_MATRIX_FREE (Chandle) ;                                              \
}
#endif

//------------------------------------------------------------------------------
// template code for C=A*B via the saxpy3 method
//------------------------------------------------------------------------------

{
    double tic = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Cp = C->p ;
    // const int64_t *GB_RESTRICT Ch = C->h ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnvec = C->nvec ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const GB_BTYPE *GB_RESTRICT Bx = B_is_pattern ? NULL : B->x ;
    // const int64_t bvlen = B->vlen ;
    // const int64_t bnvec = B->nvec ;
    const bool B_is_hyper = B->is_hyper ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const bool A_is_hyper = GB_IS_HYPER (A) ;
    const GB_ATYPE *GB_RESTRICT Ax = A_is_pattern ? NULL : A->x ;

    const int64_t *GB_RESTRICT Mp = NULL ;
    const int64_t *GB_RESTRICT Mh = NULL ;
    const int64_t *GB_RESTRICT Mi = NULL ;
    const GB_void *GB_RESTRICT Mx = NULL ;
    size_t msize = 0 ;
    int64_t mnvec = 0 ;
    bool M_is_hyper ;
    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mi = M->i ;
        Mx = (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
        mnvec = M->nvec ;
        M_is_hyper = M->is_hyper ;
    }

    //==========================================================================
    // phase0: scatter M for fine tasks
    //==========================================================================

//   printf ("phase0\n") ;

    // 3 cases:
    //      M not present and Mask_comp false: compute C=A*B
    //      M present     and Mask_comp false: compute C<M>=A*B
    //      M present     and Mask_comp true : compute C<!M>=A*B
    // If M is NULL on input, then Mask_comp is also false on input.

    bool mask_is_M = (M != NULL && !Mask_comp) ;

    // phase0 does not depend on Mask_comp, nor does it depend on the semiring
    // or data types of C, A, and B.

    // At this point, all of Hf [...] is zero, for all tasks.
    // Hi and Hx are not initialized.

    // TODO: move phase1 for coarse tasks to this phase0.
    // Then phase0 does not depend on the data type, and can be made
    // its own function used by all semirings.  Then use modular 
    // arithmetic to put coarse tasks first (fine tasks will be very
    // fast in phase0).

    int taskid ;

    if (M != NULL)
    {
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (taskid = 0 ; taskid < nfine ; taskid++)
        {

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            int64_t kk = TaskList [taskid].vector ;
            int64_t hash_size = TaskList [taskid].hsize ;
            bool use_Gustavson = (hash_size == cvlen) ;

            // partition M(:,j)
            GB_GET_M_j ;        // get M(:,j)
            int team_size = TaskList [taskid].team_size ;
            int master    = TaskList [taskid].master ;
            int my_teamid = taskid - master ;
            int64_t mystart, myend ;
            GB_PARTITION (mystart, myend, mjnz, my_teamid, team_size) ;
            mystart += pM ;
            myend   += pM ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase0: fine Gustavson task, C<M>=A*B or C<!M>=A*B
                //--------------------------------------------------------------

                // Scatter the values of M(:,j) into Hf.  No atomics needed
                // since all indices i in M(;,j) are unique.

//              printf ("%3d phase0: fine Gustavson\n", taskid) ;
                uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                for (int64_t p = mystart ; p < myend ; p++) // scan my M(:,j)
                { 
                    GB_GET_M_ij (p) ;               // get M(i,j)
                    if (mij) Hf [Mi [p]] = 1 ;      // Hf [i] = M(i,j)
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase0: fine hash task, C<M>=A*B or C<!M>=A*B
                //--------------------------------------------------------------

                // The least significant 2 bits of Hf [hash] is the flag f, and
                // the upper bits contain h, as (h,f).  After this phase0, if
                // M(i,j)=1 then the hash table contains ((i+1),1) in Hf [hash]
                // at some location.

                // Later, the flag values of f = 2 and 3 are also used.
                // Only f=1 is set in this phase.

                // h == 0,   f == 0: unoccupied and unlocked
                // h == i+1, f == 1: occupied with M(i,j)=1

//              printf ("%3d phase0: fine hash\n", taskid) ;
                int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                int64_t hash_bits = (hash_size-1) ;
                for (int64_t p = mystart ; p < myend ; p++) // scan my M(:,j)
                {
                    GB_GET_M_ij (p) ;               // get M(i,j)
                    if (!mij) continue ;            // skip if M(i,j)=0
                    int64_t i = Mi [p] ;
                    int64_t i_mine = ((i+1) << 2) + 1 ;  // ((i+1),1)
                    for (GB_HASH (i))
                    {
                        int64_t hf ;
                        // swap my hash entry into the hash table
                        #pragma omp atomic capture
                        {
                            hf = Hf [hash] ; Hf [hash] = i_mine ;
                        }
                        if (hf == 0) break ;        // success
                        // i_mine has been inserted, but a prior entry was
                        // already there.  It needs to be replaced, so take
                        // ownership of this displaced entry, and keep
                        // looking until a new empty slot is found for it.
                        i_mine = hf ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // check result for phase0
        //----------------------------------------------------------------------

        #ifdef GB_DEBUG
        for (taskid = 0 ; taskid < nfine ; taskid++)
        {
            int64_t kk = TaskList [taskid].vector ;
            ASSERT (kk >= 0 && kk < cvlen) ;
            int64_t hash_size = TaskList [taskid].hsize ;
            bool use_Gustavson = (hash_size == cvlen) ;
            int master = TaskList [taskid].master ;
            if (master != taskid) continue ;
            GB_GET_M_j ;        // get M(:,j)
            int64_t mjcount2 = 0 ;
            int64_t mjcount = 0 ;
            for (int64_t p = pM ; p < pM_end ; p++)
            {
                GB_GET_M_ij (p) ;               // get M(i,j)
                if (mij) mjcount++ ;
            }
            if (use_Gustavson)
            {
                // phase0: fine Gustavson task, C<M>=A*B or C<!M>=A*B
                uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                for ( ; pM < pM_end ; pM++)
                { 
                    GB_GET_M_ij (pM) ;               // get M(i,j)
                    ASSERT (Hf [Mi [pM]] == mij) ;
                }
                for (int64_t i = 0 ; i < cvlen ; i++)
                {
                    ASSERT (Hf [i] == 0 || Hf [i] == 1) ;
                    if (Hf [i] == 1) mjcount2++ ;
                }
                ASSERT (mjcount == mjcount2) ;
            }
            else
            {
                // phase0: fine hash task, C<M>=A*B or C<!M>=A*B
                // h == 0,   f == 0: unoccupied and unlocked
                // h == i+1, f == 1: occupied with M(i,j)=1
                int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                int64_t hash_bits = (hash_size-1) ;
                for ( ; pM < pM_end ; pM++)
                {
                    GB_GET_M_ij (pM) ;              // get M(i,j)
                    if (!mij) continue ;            // skip if M(i,j)=0
                    int64_t i = Mi [pM] ;
                    int64_t i_mine = ((i+1) << 2) + 1 ;  // ((i+1),1)
                    int64_t probe = 0 ;
                    for (GB_HASH (i))
                    {
                        int64_t hf = Hf [hash] ;
                        if (hf == i_mine) { mjcount2++ ; break ; }
                        ASSERT (hf != 0) ;
                        probe++ ;
                        ASSERT (probe < cvlen) ;
                    }
                }
                ASSERT (mjcount == mjcount2) ;
                mjcount2 = 0 ;
                for (int64_t hash = 0 ; hash < hash_size ; hash++)
                {
                    int64_t hf = Hf [hash] ;
                    int64_t h = (hf >> 2) ;     // empty (0), or a 1-based 
                    int64_t f = (hf & 3) ;      // 0 if empty or 1 if occupied
                    if (f == 1) ASSERT (h >= 1 && h <= cvlen) ;
                    ASSERT (hf == 0 || f == 1) ;
                    if (f == 1) mjcount2++ ;
                }
                ASSERT (mjcount == mjcount2) ;
            }
        }
        #endif
    }

    tic = omp_get_wtime ( ) - tic ;
    printf ("phase0: %g\n", tic) ;
    tic = omp_get_wtime ( ) ;

    //==========================================================================
    // phase1: count nnz(C(:,j)) for all tasks; do numeric work for fine tasks
    //==========================================================================

//  printf ("phase1\n") ;

    // Coarse tasks: compute nnz (C(:,kfirst:klast)).
    // Fine tasks: compute nnz (C(:,j)), and values in Hx via atomics.

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kk = TaskList [taskid].vector ;
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        bool is_fine = (kk >= 0) ;      // TODO use this test: (taskid < nfine)

        if (is_fine)
        {

            //------------------------------------------------------------------
            // fine task: compute nnz (C(:,j)) and values in Hx
            //------------------------------------------------------------------

            int64_t pB     = TaskList [taskid].start ;
            int64_t pB_end = TaskList [taskid].end + 1 ;
            #if !GB_IS_ANY_PAIR_SEMIRING
            GB_CTYPE *GB_RESTRICT Hx = TaskList [taskid].Hx ;
            #endif
            int64_t pleft = 0, pright = anvec-1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase1a: fine Gustavson task
                //--------------------------------------------------------------

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

                uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase1a: fine Gustavson task, C=A*B
                    //----------------------------------------------------------

                    // Hf [i] is initially 0.

                    // 0 -> 3 : to lock, if i seen for first time
                    // 2 -> 3 : to lock, if i seen already
                    // 3 -> 2 : to unlock; now i has been seen

//                  printf ("%3d phase1: fine Gustavson C=A*B\n", taskid) ;
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        if (aknz == 0) continue ;
                        GB_GET_B_kj ;               // bkj = B(k,j)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;   // get A(i,k)
                            GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                            uint8_t f ;

                            #if GB_IS_ANY_MONOID

                                // #pragma omp atomic read
                                f = Hf [i] ;            // grab the entry
                                if (f == 2) continue ;      // check if already updated
                                // #pragma omp atomic write
                                Hf [i] = 2 ;                // flag the entry
                                GB_ATOMIC_WRITE (i, t) ;    // Hx [i] = t

                            #else

                            #if GB_HAS_ATOMIC
                            #pragma omp atomic read
                            f = Hf [i] ;            // grab the entry
                            if (f == 2)             // if true, update C(i,j)
                            { 
                                GB_ATOMIC_UPDATE (i, t) ;   // Hx [i] += t
                                continue ;          // C(i,j) has been updated
                            }
                            #endif
                            do  // lock the entry
                            {
                                #pragma omp atomic capture
                                {
                                    f = Hf [i] ; Hf [i] = 3 ;
                                }
                            } while (f == 3) ; // lock owner gets f=0 or 2
                            bool first_time = (f == 0) ;
                            if (first_time)
                            { 
                                // C(i,j) is a new entry
                                GB_ATOMIC_WRITE (i, t) ;    // Hx [i] = t
                            }
                            else // f == 2
                            { 
                                // C(i,j) already appears in C(:,j)
                                GB_ATOMIC_UPDATE (i, t) ;   // Hx [i] += t
                            }
                            #pragma omp atomic write
                            Hf [i] = 2 ;                // unlock the entry

                            #endif

                        }
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase1a: fine Gustavson task, C<M>=A*B
                    //----------------------------------------------------------

                    // Hf [i] is 0 if M(i,j) not present or M(i,j)=0.
                    // 0 -> 1 : has already been done in phase0 if M(i,j)=1

                    // 0 -> 0 : to ignore, if M(i,j)=0
                    // 1 -> 3 : to lock, if i seen for first time
                    // 2 -> 3 : to lock, if i seen already
                    // 3 -> 2 : to unlock; now i has been seen

//                  printf ("%3d phase1: fine Gustavson C=<M>A*B\n", taskid) ;
                    GB_GET_M_j ;                // get M(:,j)
                    GB_GET_M_j_RANGE (16) ;     // get first and last in M(:,j)
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
                        GB_GET_B_kj ;               // bkj = B(k,j)

                        #if GB_IS_ANY_MONOID

                        #define GB_IKJ                                         \
                            uint8_t f ;                                        \
                            /* GB_PRAGMA (omp atomic read) */                  \
                            f = Hf [i] ;            /* grab the entry */       \
                            if (f == 0 || f == 2) continue ;                   \
                            /* GB_PRAGMA (omp atomic write) */                 \
                            Hf [i] = 2 ;                /* unlock the entry */ \
                            GB_MULT_A_ik_B_kj ;     /* t = A(i,k) * B(k,j) */  \
                            GB_ATOMIC_WRITE (i, t) ;    /* Hx [i] = t */   \

                        #else

                        #define GB_IKJ                                         \
                        {                                                      \
                            GB_MULT_A_ik_B_kj ;     /* t = A(i,k) * B(k,j) */  \
                            uint8_t f ;                                        \
                            GB_PRAGMA (omp atomic read)                        \
                            f = Hf [i] ;            /* grab the entry */       \
                            if (GB_HAS_ATOMIC && (f == 2))                     \
                            {                                                  \
                                /* C(i,j) already seen; update it */           \
                                GB_ATOMIC_UPDATE (i, t) ;   /* Hx [i] += t */  \
                                continue ;       /* C(i,j) has been updated */ \
                            }                                                  \
                            if (f == 0) continue ; /* M(i,j)=0; ignore C(i,j)*/\
                            do  /* lock the entry */                           \
                            {                                                  \
                                GB_PRAGMA (omp atomic capture)                 \
                                {                                              \
                                    f = Hf [i] ; Hf [i] = 3 ;                  \
                                }                                              \
                            } while (f == 3) ; /* lock owner gets f=1 or 2 */  \
                            bool first_time = (f == 1) ;                       \
                            if (first_time)                                    \
                            {                                                  \
                                /* C(i,j) is a new entry */                    \
                                GB_ATOMIC_WRITE (i, t) ;    /* Hx [i] = t */   \
                            }                                                  \
                            else /* f == 2 */                                  \
                            {                                                  \
                                /* C(i,j) already appears in C(:,j) */         \
                                GB_ATOMIC_UPDATE (i, t) ;   /* Hx [i] += t */  \
                            }                                                  \
                            GB_PRAGMA (omp atomic write)                       \
                            Hf [i] = 2 ;                /* unlock the entry */ \
                        }
                        #endif

                        GB_SCAN_M_j_OR_A_k ;
                        #undef GB_IKJ
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase1a: fine Gustavson task, C<!M>=A*B
                    //----------------------------------------------------------

                    // Hf [i] is 0 if M(i,j) not present or M(i,j)=0.
                    // 0 -> 1 : has already been done in phase0 if M(i,j)=1

                    // 1 -> 1 : to ignore, if M(i,j)=1
                    // 0 -> 3 : to lock, if i seen for first time
                    // 2 -> 3 : to lock, if i seen already
                    // 3 -> 2 : to unlock; now i has been seen

//                  printf ("%3d phase1: fine Gustavson C=<!M>A*B\n", taskid) ;
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        if (aknz == 0) continue ;
                        GB_GET_B_kj ;               // bkj = B(k,j)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;   // get A(i,k)
                            GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                            uint8_t f ;

                            #if GB_IS_ANY_MONOID

                            // #pragma omp atomic read
                            f = Hf [i] ;            // grab the entry
                            if (f == 1 || f == 2) continue ;
                            // #pragma omp atomic write
                            Hf [i] = 2 ;                // unlock the entry
                            GB_ATOMIC_WRITE (i, t) ;    // Hx [i] = t

                            #else

                            GB_MULT_A_ik_B_kj ;     // t = A(i,k) * B(k,j)
                            #pragma omp atomic read
                            f = Hf [i] ;            // grab the entry
                            #if GB_HAS_ATOMIC
                            if (f == 2)             // if true, update C(i,j)
                            { 
                                GB_ATOMIC_UPDATE (i, t) ;   // Hx [i] += t
                                continue ;          // C(i,j) has been updated
                            }
                            #endif
                            if (f == 1) continue ; // M(i,j)=1; ignore C(i,j)
                            do  // lock the entry
                            {
                                #pragma omp atomic capture
                                {
                                    f = Hf [i] ; Hf [i] = 3 ;
                                }
                            } while (f == 3) ; // lock owner of gets f=0 or 2
                            bool first_time = (f == 0) ;
                            if (first_time)
                            { 
                                // C(i,j) is a new entry
                                GB_ATOMIC_WRITE (i, t) ;    // Hx [i] = t
                            }
                            else // f == 2
                            { 
                                // C(i,j) already seen
                                GB_ATOMIC_UPDATE (i, t) ;   // Hx [i] += t
                            }
                            #pragma omp atomic write
                            Hf [i] = 2 ;                // unlock the entry
                            #endif
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase1: fine hash task
                //--------------------------------------------------------------

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

                int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                int64_t hash_bits = (hash_size-1) ;
                int64_t my_cjnz = 0 ; // TODO count later?

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase1: fine hash task, C=A*B
                    //----------------------------------------------------------

                    // Given Hf [hash] split into (h,f)

                    // h == 0  , f == 0 : unlocked and unoccupied.
                    // h == i+1, f == 2 : unlocked, occupied by C(i,j).
                    //                    Hx is initialized.
                    // h == ..., f == 3 : locked.

                    // 0 -> 3 : to lock, if i seen for first time
                    // 2 -> 3 : to lock, if i seen already
                    // 3 -> 2 : to unlock; now i has been seen

//                  printf ("%3d phase1: fine hash C=A*B\n", taskid) ;
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        if (aknz == 0) continue ;
                        GB_GET_B_kj ;               // bkj = B(k,j)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                            int64_t i1 = i + 1 ;        // i1 = one-based index
                            int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)
                            for (GB_HASH (i))           // find i in hash table
                            {
                                int64_t hf ;
                                #pragma omp atomic read
                                hf = Hf [hash] ;        // grab the entry
                                #if GB_HAS_ATOMIC
                                if (hf == i_unlocked)  // if true, update C(i,j)
                                { 
                                    GB_ATOMIC_UPDATE (hash, t) ;// Hx [hash]+=t
                                    break ;         // C(i,j) has been updated
                                }
                                #endif
                                int64_t h = (hf >> 2) ;
                                if (h == 0 || h == i1)
                                {
                                    // h=0: unoccupied, h=i1: occupied by i
                                    do  // lock the entry
                                    {
                                        #pragma omp atomic capture
                                        {
                                            hf = Hf [hash] ; Hf [hash] |= 3 ;
                                        }
                                    } while ((hf & 3) == 3) ; // owner: f=0 or 2
                                    bool first_time = (hf == 0) ;
                                    if (first_time) // f == 0
                                    { 
                                        // C(i,j) is a new entry in C(:,j)
                                        // Hx [hash] = t
                                        GB_ATOMIC_WRITE (hash, t) ;
                                        #pragma omp atomic write
                                        Hf [hash] = i_unlocked ; // unlock entry
                                        my_cjnz++ ; // TODO count later?
                                        break ;
                                    }
                                    if (hf == i_unlocked) // f == 2
                                    { 
                                        // C(i,j) already appears in C(:,j)
                                        // Hx [hash] += t
                                        GB_ATOMIC_UPDATE (hash, t) ;
                                        #pragma omp atomic write
                                        Hf [hash] = i_unlocked ; // unlock entry
                                        break ;
                                    }
                                    // hash table occupied, but not with i
                                    #pragma omp atomic write
                                    Hf [hash] = hf ;  // unlock with prior value
                                }
                            }
                        }
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase1: fine hash task, C<M>=A*B
                    //----------------------------------------------------------

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

//                  printf ("%3d phase1: fine hash C<M>=A*B\n", taskid) ;
                    GB_GET_M_j ;                // get M(:,j)
                    GB_GET_M_j_RANGE (16) ;     // get first and last in M(:,j)
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
                        GB_GET_B_kj ;               // bkj = B(k,j)
                        #define GB_IKJ                                         \
                        {                                                      \
                            GB_MULT_A_ik_B_kj ;      /* t = A(i,k) * B(k,j) */ \
                            int64_t i1 = i + 1 ;     /* i1 = one-based index */\
                            int64_t i_unlocked = (i1 << 2) + 2 ;  /* (i+1,2) */\
                            for (GB_HASH (i))        /* find i in hash table */\
                            {                                                  \
                                int64_t hf ;                                   \
                                GB_PRAGMA (omp atomic read)                    \
                                hf = Hf [hash] ;        /* grab the entry */   \
                                if (GB_HAS_ATOMIC && (hf == i_unlocked))       \
                                {                                              \
                                    GB_ATOMIC_UPDATE (hash, t) ;/* Hx [.]+=t */\
                                    break ;     /* C(i,j) has been updated */  \
                                }                                              \
                                if (hf == 0) break ; /* M(i,j)=0; ignore Cij */\
                                if ((hf >> 2) == i1) /* if true, i found */    \
                                {                                              \
                                    do /* lock the entry */                    \
                                    {                                          \
                                        GB_PRAGMA (omp atomic capture)         \
                                        {                                      \
                                            hf = Hf [hash] ; Hf [hash] |= 3 ;  \
                                        }                                      \
                                    } while ((hf & 3) == 3) ; /* own: f=1,2 */ \
                                    bool first_time = ((hf & 3) == 1) ;        \
                                    if (first_time) /* f == 1 */               \
                                    {                                          \
                                        /* C(i,j) is a new entry in C(:,j) */  \
                                        /* Hx [hash] = t */                    \
                                        GB_ATOMIC_WRITE (hash, t) ;            \
                                    }                                          \
                                    else /* f == 2 */                          \
                                    {                                          \
                                        /* C(i,j) already appears in C(:,j) */ \
                                        /* Hx [hash] += t */                   \
                                        GB_ATOMIC_UPDATE (hash, t) ;           \
                                    }                                          \
                                    GB_PRAGMA (omp atomic write)               \
                                    Hf [hash] = i_unlocked ; /* unlock entry */\
                                    if (first_time) my_cjnz++ ; /* TODO count later? */ \
                                    break ;                                    \
                                }                                              \
                            }                                                  \
                        }
                        GB_SCAN_M_j_OR_A_k ;
                        #undef GB_IKJ
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase1: fine hash task, C<!M>=A*B
                    //----------------------------------------------------------

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

//                  printf ("%3d phase1: fine hash C<!M>=A*B\n", taskid) ;
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        if (aknz == 0) continue ;
                        GB_GET_B_kj ;               // bkj = B(k,j)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            GB_MULT_A_ik_B_kj ;         // t = A(i,k) * B(k,j)
                            int64_t i1 = i + 1 ;        // i1 = one-based index
                            int64_t i_unlocked = (i1 << 2) + 2 ;    // (i+1,2)
                            int64_t i_masked   = (i1 << 2) + 1 ;    // (i+1,1)
                            for (GB_HASH (i))           // find i in hash table
                            {
                                int64_t hf ;
                                #pragma omp atomic read
                                hf = Hf [hash] ;        // grab the entry
                                #if GB_HAS_ATOMIC
                                if (hf == i_unlocked)  // if true, update C(i,j)
                                { 
                                    GB_ATOMIC_UPDATE (hash, t) ;// Hx [hash]+=t
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
                                        #pragma omp atomic capture
                                        {
                                            hf = Hf [hash] ; Hf [hash] |= 3 ;
                                        }
                                    } while ((hf & 3) == 3) ; // owner: f=0,1,2
                                    bool first_time = (hf == 0) ;
                                    if (first_time)    // f == 0
                                    { 
                                        // C(i,j) is a new entry in C(:,j)
                                        // Hx [hash] = t
                                        GB_ATOMIC_WRITE (hash, t) ;
                                        #pragma omp atomic write
                                        Hf [hash] = i_unlocked ; // unlock entry
                                        my_cjnz++ ; // TODO count later?
                                        break ;
                                    }
                                    if (hf == i_unlocked)   // f == 2
                                    { 
                                        // C(i,j) already appears in C(:,j)
                                        // Hx [hash] += t
                                        GB_ATOMIC_UPDATE (hash, t) ;
                                        #pragma omp atomic write
                                        Hf [hash] = i_unlocked ; // unlock entry
                                        break ;
                                    }
                                    // hash table occupied, but not with i,
                                    // or with i but M(i,j)=1 so C(i,j) ignored
                                    #pragma omp atomic write
                                    Hf [hash] = hf ;  // unlock with prior value
                                }
                            }
                        }
                    }
                }

                TaskList [taskid].my_cjnz = my_cjnz ;   // TODO count later?
            }


        }
        else
        {

            //------------------------------------------------------------------
            // coarse tasks: compute nnz in each vector of A*B(:,kfirst:klast)
            //------------------------------------------------------------------

            int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            int64_t kfirst = TaskList [taskid].start ;
            int64_t klast  = TaskList [taskid].end ;
            int64_t mark = 0 ;
            // int64_t nk = klast - kfirst + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase1: coarse Gustavson task
                //--------------------------------------------------------------

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase1: coarse Gustavson task, C=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all Hf.
                    // Hf [i] is set to mark when C(i,j) is found.

//                  printf ("%3d phase1: coarse Gustavson C=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                        if (bjnz == 1)
                        { 
                            int64_t k = Bi [pB] ;   // get B(k,j)
                            GB_GET_A_k ;            // get A(:,k)
                            Cp [kk] = aknz ;        // nnz(C(:,j)) = nnz(A(:,k))
                            continue ;
                        }
                        mark++ ;
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == cvlen)
                            { 
                                cjnz = cvlen ;  // A(:,k) is dense
                                break ;         // so nnz(C(:,j)) = cvlen
                            }
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k)
                                if (Hf [i] != mark) // if true, i is new
                                { 
                                    Hf [i] = mark ; // mark C(i,j) as seen
                                    cjnz++ ;        // C(i,j) is a new entry
                                }
                            }
                        }
                        Cp [kk] = cjnz ;    // count the entries in C(:,j)
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase1: coarse Gustavson task, C<M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.

                    // Hf [i] < mark    : M(i,j)=0, C(i,j) is ignored.
                    // Hf [i] == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // Hf [i] == mark+1 : M(i,j)=1, and C(i,j) has been seen.

//                  printf ("%3d phase1: coarse Gustavson C<M>=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                        GB_GET_M_j ;            // get M(:,j)
                        if (mjnz == 0) { Cp [kk] = 0 ; continue ; }
                        GB_GET_M_j_RANGE (64) ; // get first and last in M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_SCATTER_M_j ;        // scatter M(:,j)
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
                            #define GB_IKJ                                     \
                            {                                                  \
                                if (Hf [i] == mark)   /* if true, M(i,j) is 1*/\
                                {                                              \
                                    Hf [i] = mark1 ;  /* mark C(i,j) as seen */\
                                    cjnz++ ;          /* C(i,j) is new */      \
                                }                                              \
                            }
                            GB_SCAN_M_j_OR_A_k ;
                            #undef GB_IKJ
                        }
                        Cp [kk] = cjnz ;    // count the entries in C(:,j)
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase1: coarse Gustavson task, C<!M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.

                    // Hf [i] < mark    : M(i,j)=0, C(i,j) is not yet seen.
                    // Hf [i] == mark   : M(i,j)=1, so C(i,j) is ignored.
                    // Hf [i] == mark+1 : M(i,j)=0, and C(i,j) has been seen.

//                  printf ("%3d phase1: coarse Gustavson C<!M>=A*B\n", taskid);
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;                    // get B(:,j)
                        if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                        GB_GET_M_j ;            // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_SCATTER_M_j ;        // scatter M(:,j)
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k)
                                if (Hf [i] < mark)      // if true, M(i,j) is 0
                                { 
                                    Hf [i] = mark1 ;    // mark C(i,j) as seen
                                    cjnz++ ;            // C(i,j) is a new entry
                                }
                            }
                        }
                        Cp [kk] = cjnz ;    // count the entries in C(:,j)
                    }
                }

            }

            #if 0
            else if (nk == 1)
            {

                //--------------------------------------------------------------
                // phase1: 1-vector coarse hash task (TODO: M or !M broken)
                //--------------------------------------------------------------

/* without this method, single-thread performance can suffer.  The results
   below are likely using coarse Gustavson, but might be faster with
   the 1-thread coarse hash task.  Note the super-linear speedup.

slash $ ./build/bfs_test ~/com-Orkut.grb 
chunk: 65536
threads:  1 2 3 4 7 8
matrix: /home/davis/com-Orkut.grb
[.grb]
Reading binary file: /home/davis/com-Orkut.grb
read time: 1.22315

A is symmetric
transpose time: 12.3188

==========input graph: nodes: 3072441 edges: 234370166 ntrials: 64
allpush (with tree):
: 1:allpush   (w/ tree):        4.457 (sec), rate:  52.59
: 2:allpush   (w/ tree):        0.929 (sec), rate: 252.20 speedup  4.8
: 3:allpush   (w/ tree):        0.727 (sec), rate: 322.36 speedup  6.1
: 4:allpush   (w/ tree):        0.621 (sec), rate: 377.35 speedup  7.2
: 7:allpush   (w/ tree):        0.557 (sec), rate: 420.92 speedup  8.0
: 8:allpush   (w/ tree):        0.551 (sec), rate: 425.11 speedup  8.1
*/


                // The least 2 bits of Hf [hash] are used for f, and the upper
                // bits are used for h = i+1, the 1-based index for entry i.

                int64_t hash_bits = (hash_size-1) ;
                int64_t kk = kfirst ;
                int64_t cjnz = 0 ;

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase1: 1-vector coarse hash task, C=A*B
                    //----------------------------------------------------------

                    // Hf is initially all zero.

                    // h == 0,   f == 0 : unoccupied
                    // h == i+1, f == 2 : C(i,j) has been seen in phase1

//                  printf ("%3d phase1: 1vec coarse hash C=A*B\n", taskid) ;
                    GB_GET_B_j ;            // get B(:,j)
                    if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                    if (bjnz == 1)
                    { 
                        int64_t k = Bi [pB] ;   // get B(k,j)
                        GB_GET_A_k ;            // get A(:,k)
                        Cp [kk] = aknz ;        // nnz(C(:,j)) = nnz(A(:,k))
                        continue ;
                    }
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;   // get A(i,k)
                            int64_t i_seen = ((i+1) << 2) + 2 ;
                            for (GB_HASH (i))       // find i in hash table
                            {
                                int64_t h = Hf [hash] ;
                                if (h == i_seen) break ;  // already seen
                                if (h == 0)           // if true, i is new
                                { 
                                    Hf [hash] = i_seen ;  // mark as seen
                                    cjnz++ ;          // C(i,j) is new entry
                                    break ;
                                }
                            }
                        }
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase1: 1-vector coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    // Hf is initially all zero.  This task uses Hf [hash] in a
                    // similar way as the fine hash task, except that the
                    // locked state (h,3) is not used.

                    // h == 0  , f == 0 : unoccupied.
                    // h == i+1, f == 1 : occupied by M(i,j)=1. C(i,j) not seen.
                    // h == i+1, f == 2 : M(i,j)=1, and C(i,j) seen in phase1

//                  printf ("%3d phase1: 1vec coarse hash C<M>=A*B\n",taskid) ;
                    GB_GET_B_j ;                // get B(:,j)
                    if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                    GB_GET_M_j ;                // get M(:,j)
                    if (mjnz == 0) { Cp [kk] = 0 ; continue ; }
                    GB_GET_M_j_RANGE ;          // get first and last in M(:,j)
                    GB_HASH1_M_j ;              // hash M(:,j)
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            int64_t i1 = i + 1 ;
                            int64_t i_mark = (i1 << 2) + 1 ;    // (i+1,1)
                            int64_t i_seen = (i1 << 2) + 2 ;    // (i+2,2)
                            for (GB_HASH (i))       // find i in hash table
                            {
                                int64_t hf = Hf [hash] ;
                                int64_t h = (hf >> 2) ;
                                if (h == 0) break ;     // M(i,j)=0, ignore i
                                if (h == i1) break ;    // i already seen
                                if (hf == i_mark)       // if true, M(i,j) is 1
                                {
                                    Hf [hash] = i_seen ;  // mark C(i,j) as seen
                                    cjnz++ ;              // C(i,j) is new entry
                                    break ;
                                }
                            }
                        }
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase1: 1-vector coarse hash task, C<!M>=A*B
                    //----------------------------------------------------------

                    // Hf is initially all zero.  This task uses Hf [hash] in a
                    // similar way as the fine hash task, except that the
                    // locked state (h,3) is not used.

                    // h == 0  , f == 0 : unoccupied.
                    // h == i+1, f == 1 : occupied by M(i,j)=1. C(i,j) ignored.
                    // h == i+1, f == 2 : M(i,j)=0, and C(i,j) seen in phase1

//                  printf ("%3d phase1: 1vec coarse hash C<!M>=A*B\n",taskid);
                    GB_GET_B_j ;                // get B(:,j)
                    if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                    GB_GET_M_j ;                // get M(:,j)
                    GB_HASH1_M_j ;              // hash M(:,j)
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            int64_t i1 = i + 1 ;
                            int64_t i_seen = (i1 << 2) + 2 ;    // (i+2,2)
                            for (GB_HASH (i))       // find i in hash table
                            {
                                int64_t hf = Hf [hash] ;
                                int64_t h = (hf >> 2) ;
                                if (h == i1) break ; // i seen, or ignored
                                if (h == 0)          // if true, M(i,j) is 0
                                { 
                                    Hf [hash] = i_seen ;  // mark C(i,j) as seen
                                    cjnz++ ;              // C(i,j) is new entry
                                    break ;
                                }
                            }
                        }
                    }
                }

                //--------------------------------------------------------------
                // save nnz(C(:,j)) for all 1-vector coarse hash tasks
                //--------------------------------------------------------------

                Cp [kk] = cjnz ;    // count the entries in C(:,j)

            }
            #endif
            else
            {

                //--------------------------------------------------------------
                // phase1: multi-vector coarse hash task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;
                int64_t hash_bits = (hash_size-1) ;

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase1: multi-vector coarse hash task, C=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let f = Hf [hash] and h = Hi [hash]

                    // f < mark          : unoccupied.
                    // h == i, f == mark : occupied with C(i,j)

//                  printf ("%3d phase1: nvec coarse hash C=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                        if (bjnz == 1)
                        { 
                            int64_t k = Bi [pB] ;   // get B(k,j)
                            GB_GET_A_k ;            // get A(:,k)
                            Cp [kk] = aknz ;        // nnz(C(:,j)) = nnz(A(:,k))
                            continue ;
                        }
                        mark++ ;
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k)
                                for (GB_HASH (i))       // find i in hash
                                {
                                    if (Hf [hash] < mark)
                                    { 
                                        Hf [hash] = mark ; // insert C(i,j)
                                        Hi [hash] = i ;
                                        cjnz++ ;  // C(i,j) is a new entry.
                                        break ;
                                    }
                                    if (Hi [hash] == i) break ;
                                }
                            }
                        }
                        Cp [kk] = cjnz ;    // count the entries in C(:,j)
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase1: multi-vector coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark: unoccupied, M(i,j)=0, C(i,j) ignored if
                    //           this case occurs while scanning A(:,k)
                    // h == i, f == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // h == i, f == mark+1 : M(i,j)=1, and C(i,j) has been seen.

//                  printf ("%3d phase1: nvec coarse hash C<M>=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                        GB_GET_M_j ;            // get M(:,j)
                        if (mjnz == 0) { Cp [kk] = 0 ; continue ; }
                        GB_GET_M_j_RANGE (64) ; // get first and last in M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;           // hash M(:,j)
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
                            #define GB_IKJ                                     \
                            {                                                  \
                                for (GB_HASH (i))       /* find i in hash */   \
                                {                                              \
                                    int64_t f = Hf [hash] ;                    \
                                    if (f < mark) break ; /* M(i,j)=0; ignore*/\
                                    if (Hi [hash] == i)   /* if true, i found*/\
                                    {                                          \
                                        if (f == mark)  /* if true, i is new */\
                                        {                                      \
                                            Hf [hash] = mark1 ; /* mark seen */\
                                            cjnz++ ;    /* C(i,j) is new */    \
                                        }                                      \
                                        break ;                                \
                                    }                                          \
                                }                                              \
                            }
                            GB_SCAN_M_j_OR_A_k ;
                            #undef GB_IKJ
                        }
                        Cp [kk] = cjnz ;    // count the entries in C(:,j)
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase1: multi-vector coarse hash task, C<!M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark: unoccupied, M(i,j)=0, and C(i,j) not yet seen.
                    // h == i, f == mark   : M(i,j)=1. C(i,j) ignored.
                    // h == i, f == mark+1 : M(i,j)=0, and C(i,j) has been seen.

//                  printf("%3d phase1: #vec coarse hash C<!M>=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0) { Cp [kk] = 0 ; continue ; }
                        GB_GET_M_j ;            // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;           // hash M(:,j)
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k)
                                for (GB_HASH (i))       // find i in hash
                                {
                                    if (Hf [hash] < mark)   // if true, i is new
                                    { 
                                        Hf [hash] = mark1 ; // mark C(i,j) seen
                                        Hi [hash] = i ;
                                        cjnz++ ;        // C(i,j) is a new entry
                                        break ;
                                    }
                                    if (Hi [hash] == i) break ;
                                }
                            }
                        }
                        Cp [kk] = cjnz ;    // count the entries in C(:,j)
                    }
                }
            }
        }
    }

    tic = omp_get_wtime ( ) - tic ;
    printf ("phase1a: %g\n", tic) ;
    tic = omp_get_wtime ( ) ;


    // phase1b

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kk = TaskList [taskid].vector ;
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        bool is_fine = (kk >= 0) ;      // TODO use this test: (taskid < nfine)

        if (is_fine)
        {

            //------------------------------------------------------------------
            // fine task: compute nnz (C(:,j)) and values in Hx
            //------------------------------------------------------------------

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase1b: fine Gustavson task, C=A*B, C<M>=A*B, or C<!M>=A*B
                //--------------------------------------------------------------

                // Hf [i] == 2 if C(i,j) is an entry in C(:,j)

                int team_size = TaskList [taskid].team_size ;
                int master    = TaskList [taskid].master ;
                int my_teamid = taskid - master ;
                int64_t my_cjnz = 0 ;

                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cvlen, my_teamid, team_size) ;

                // C(:,j) is sparse: count the work for this fine task
                uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                // scan the hash table for entries in C(:,j)

                for (int64_t i = istart ; i < iend ; i++)
                {
                    if (Hf [i] == 2)
                    { 
                        my_cjnz++ ;
                    }
                }

                TaskList [taskid].my_cjnz = my_cjnz ;   // count my nnz(C(:,j))
            }
        }
    }

    tic = omp_get_wtime ( ) - tic ;
    printf ("phase1b: %g\n", tic) ;
    tic = omp_get_wtime ( ) ;

    //==========================================================================
    // phase2: compute Cp with cumulative sum and allocate C
    //==========================================================================

    // TODO make this a function, shared by all semirings.

//  printf ("phase2\n") ;

    // TaskList [taskid].my_cjnz is the # of unique entries found in C(:,j) by
    // that task.  Sum these terms to compute total # of entries in C(:,j).

    for (taskid = 0 ; taskid < nfine ; taskid++)
    { 
        int64_t kk = TaskList [taskid].vector ;
        Cp [kk] = 0 ;
    }

    for (taskid = 0 ; taskid < nfine ; taskid++)
    { 
        int64_t kk = TaskList [taskid].vector ;
        int64_t my_cjnz = TaskList [taskid].my_cjnz ;
        Cp [kk] += my_cjnz ;
        ASSERT (my_cjnz <= cvlen) ;
    }

    // Cp [kk] is now nnz (C (:,j)), for all vectors j, whether computed by
    // fine tasks or coarse tasks, and where j == (Bh == NULL) ? kk : Bh [kk].

    int nth = GB_nthreads (cnvec, chunk, nthreads) ;
    GB_cumsum (Cp, cnvec, &(C->nvec_nonempty), nth) ;
    int64_t cnz = Cp [cnvec] ;

    // allocate Ci and Cx
    GrB_Info info = GB_ix_alloc (C, cnz, true, Context) ;
    if (info != GrB_SUCCESS)
    {
        // out of memory
        GB_FREE_ALL ;
        return (info) ;
    }

    int64_t  *GB_RESTRICT Ci = C->i ;
    GB_CTYPE *GB_RESTRICT Cx = C->x ;

    #if GB_IS_ANY_PAIR_SEMIRING

        // ANY_PAIR semiring: result is purely symbolic
        int64_t pC ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (pC = 0 ; pC < cnz ; pC++)
        {
            Cx [pC] = 1 ;
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

    tic = omp_get_wtime ( ) - tic ;
    printf ("phase2: %g\n", tic) ;
    tic = omp_get_wtime ( ) ;

    //==========================================================================
    // phase3: numeric phase for coarse tasks, prep for gather for fine tasks
    //==========================================================================

//  printf ("phase3\n") ;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        #if !GB_IS_ANY_PAIR_SEMIRING
        GB_CTYPE *GB_RESTRICT Hx = TaskList [taskid].Hx ;
        #endif
        int64_t kk = TaskList [taskid].vector ;
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        bool is_fine = (kk >= 0) ;

        if (is_fine)
        {

            //------------------------------------------------------------------
            // count nnz (C(:,j)) for the final gather for this fine task
            //------------------------------------------------------------------

            int team_size = TaskList [taskid].team_size ;
            int master    = TaskList [taskid].master ;
            int my_teamid = taskid - master ;
            int64_t my_cjnz = 0 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase3: fine Gustavson task, C=A*B, C<M>=A*B, or C<!M>=A*B
                //--------------------------------------------------------------

                // Hf [i] == 2 if C(i,j) is an entry in C(:,j)

//              printf ("%3d phase3: fine Gustavson\n", taskid) ;
                int64_t pC = Cp [kk] ;
                int64_t cjnz = Cp [kk+1] - pC ;
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cvlen, my_teamid, team_size) ;
                if (cjnz == cvlen)
                {
                    // C(:,j) is entirely dense: finish the work now
                    for (int64_t i = istart ; i < iend ; i++)
                    { 
                        Ci [pC + i] = i ;
                    }
                    #if !GB_IS_ANY_PAIR_SEMIRING
                    // copy Hx [istart:iend-1] into Cx [pC+istart:pC+iend-1]
                    GB_CIJ_MEMCPY (pC + istart, istart, iend - istart) ;
                    #endif
                }

                #if 0
                else
                {

                    // C(:,j) is sparse: count the work for this fine task
                    uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                    // scan the hash table for entries in C(:,j)
                    for (int64_t i = istart ; i < iend ; i++)
                    {
                        if (Hf [i] == 2)
                        { 
                            my_cjnz++ ;
                        }
                    }

                }
                #endif

            }
            else
            {

                //--------------------------------------------------------------
                // phase3: fine hash task, C=A*B, C<M>=A*B, or C<!M>=A*B
                //--------------------------------------------------------------

                // (Hf [hash] & 3) == 2 if C(i,j) is an entry in C(:,j),
                // and the index i of the entry is (Hf [hash] >> 2) - 1.

//              printf ("%3d phase3: fine hash\n", taskid) ;
                int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                int64_t mystart, myend ;
                GB_PARTITION (mystart, myend, hash_size, my_teamid, team_size) ;
                for (int64_t hash = mystart ; hash < myend ; hash++)
                {
                    if ((Hf [hash] & 3) == 2)
                    { 
                        my_cjnz++ ;
                    }
                }

                TaskList [taskid].my_cjnz = my_cjnz ;   // count my nnz(C(:,j))
            }

        }
        else
        {

            //------------------------------------------------------------------
            // numeric coarse task: compute C(:,kfirst:klast)
            //------------------------------------------------------------------

            int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            int64_t kfirst = TaskList [taskid].start ;
            int64_t klast = TaskList [taskid].end ;
            int64_t nk = klast - kfirst + 1 ;
            int64_t mark = 2*nk + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase3: coarse Gustavson task
                //--------------------------------------------------------------

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase3: coarse Gustavson task, C=A*B
                    //----------------------------------------------------------

//                  printf ("%3d phase3: coarse Gustavson C=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;
                        int64_t cjnz = Cp [kk+1] - pC ;
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
                                int64_t k = Bi [pB] ;       // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                for ( ; pA < pA_end ; pA++) // scan A(:,k)
                                {
                                    int64_t i = Ai [pA] ;   // get A(i,k)
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
                                int64_t k = Bi [pB] ;       // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                for ( ; pA < pA_end ; pA++) // scan A(:,k)
                                {
                                    int64_t i = Ai [pA] ;   // get A(i,k)
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    if (Hf [i] != mark)
                                    { 
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [i] = mark ;
                                        GB_HX_WRITE (i, t) ;    // Hx [i] = t
                                        Ci [pC++] = i ;
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
                    // phase3: coarse Gustavson task, C<M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.

                    // Hf [i] < mark    : M(i,j)=0, C(i,j) is ignored.
                    // Hf [i] == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // Hf [i] == mark+1 : M(i,j)=1, and C(i,j) has been seen.

// HERE
//                  printf ("%3d phase3: coarse Gustavson C<M>=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;
                        int64_t cjnz = Cp [kk+1] - pC ;
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
                        GB_SCATTER_M_j ;        // scatter M(:,j)
                        if (16 * cjnz > cvlen)  // C(:,j) is not very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                int64_t k = Bi [pB] ;       // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
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
                                int64_t k = Bi [pB] ;       // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
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
                    // phase3: coarse Gustavson task, C<!M>=A*B
                    //----------------------------------------------------------

                    // if !M:
                    // Hf [i] < mark    : M(i,j)=0, C(i,j) is not yet seen.
                    // Hf [i] == mark   : M(i,j)=1, so C(i,j) is ignored.
                    // Hf [i] == mark+1 : M(i,j)=0, and C(i,j) has been seen.

//                  printf ("%3d phase3: coarse Gustavson C<!M>=A*B\n", taskid);
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;
                        int64_t cjnz = Cp [kk+1] - pC ;
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
                        GB_SCATTER_M_j ;        // scatter M(:,j)
                        if (16 * cjnz > cvlen)  // C(:,j) is not very sparse
                        {
                            for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                            {
                                int64_t k = Bi [pB] ;       // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                for ( ; pA < pA_end ; pA++) // scan A(:,k)
                                {
                                    int64_t i = Ai [pA] ;   // get A(i,k)
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
                                int64_t k = Bi [pB] ;       // get B(k,j)
                                GB_GET_A_k ;                // get A(:,k)
                                if (aknz == 0) continue ;
                                GB_GET_B_kj ;               // bkj = B(k,j)
                                for ( ; pA < pA_end ; pA++) // scan A(:,k)
                                {
                                    int64_t i = Ai [pA] ;   // get A(i,k)
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
            #if 0
            else if (nk == 1)
            {

                //--------------------------------------------------------------
                // phase3: 1-vector coarse hash task
                //--------------------------------------------------------------

                // Hf has been initialized with the pattern of C(:,j), and
                // M(:,j) for both C<M>=A*B and C<!M>=A*B.  In all three cases,
                // the only case that needs to be considered here, on input,
                // is the first one (h=i+1,f=2).

                // h == i+1, f == 2 : C(i,j) seen in phase1; Hx not initialized
                // h == i+1, f == 3 : C(i,j) seen in phase3; Hx initialized

                int64_t hash_bits = (hash_size-1) ;
                int64_t kk = kfirst ;
                int64_t pC = Cp [kk] ;
                int64_t cjnz = Cp [kk+1] - pC ;
                if (cjnz == 0) continue ;       // nothing to do
                GB_GET_B_j ;                    // get B(:,j)

                // compute the pattern and values of C(:,j)

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase3: 1-vector coarse hash task, C=A*B
                    //----------------------------------------------------------

//                  printf ("%3d phase3: 1vec coarse hash C=A*B\n", taskid) ;
                    if (bjnz == 1)          // C(:,j) = A(:,k)*B(k,j)
                    { 
                        GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE ;
                        continue ;
                    }
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        if (aknz == 0) continue ;
                        GB_GET_B_kj ;               // bkj = B(k,j)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            int64_t i1 = i + 1 ;
                            int64_t i_phase1 = (i1 << 2) + 2 ;  // (i+1,2)
                            int64_t i_phase2 = (i1 << 2) + 3 ;  // (i+1,3)
                            GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                            for (GB_HASH (i))       // find i in hash table
                            {
                                int64_t h = Hf [hash] ;
                                if (h == i_phase2)
                                { 
                                    // C(i,j) has been seen; update it.
                                    GB_HX_UPDATE (hash, t) ; // Hx [hash]+=t
                                    break ;
                                }
                                else if (h == i_phase1)
                                { 
                                    // first time C(i,j) seen here
                                    Hf [hash] = i_phase2 ;
                                    GB_HX_WRITE (hash, t) ; // Hx [hash] = t
                                    Ci [pC++] = i ;
                                    break ;
                                }
                            }
                        }
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase3: 1-vector coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

//                  printf ("%3d phase3: 1vec coarse hash C<M>=A*B\n", taskid);
                    GB_GET_B_j ;                // get B(:,j)
                    GB_GET_M_j ;                // get M(:,j)
                    GB_GET_M_j_RANGE ;          // get first and last in M(:,j)
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
                        GB_GET_B_kj ;               // bkj = B(k,j)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            int64_t i1 = i + 1 ;
                            int64_t i_phase1 = (i1 << 2) + 2 ;  // (i+1,2)
                            int64_t i_phase2 = (i1 << 2) + 3 ;  // (i+1,3)
                            for (GB_HASH (i))       // find i in hash table
                            {
                                int64_t hf = Hf [hash] ;
                                int64_t h = (hf >> 2) ;
                                if (hf == i_phase2)
                                { 
                                    // C(i,j) has been seen; update it.
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    GB_HX_UPDATE (hash, t) ;// Hx [hash] += t
                                    break ;
                                }
                                else if (hf == i_phase1)
                                { 
                                    // first time C(i,j) seen here
                                    Hf [hash] = i_phase2 ;
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    GB_HX_WRITE (hash, t) ; // Hx [hash] = t
                                    Ci [pC++] = i ;
                                    break ;
                                }
                                else if (h == 0 || h == i1) break ; // ignore i
                            }
                        }
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase3: 1-vector coarse hash task, C<!M>=A*B
                    //----------------------------------------------------------

//                  printf ("%3d phase3: 1vec coarse hash C<!M>=A*B\n",taskid);
                    GB_GET_B_j ;                // get B(:,j)
                    GB_GET_M_j ;                // get M(:,j)
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        if (aknz == 0) continue ;
                        GB_GET_B_kj ;               // bkj = B(k,j)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            int64_t i1 = i + 1 ;
                            int64_t i_phase1 = (i1 << 2) + 2 ;  // (i+1,2)
                            int64_t i_phase2 = (i1 << 2) + 3 ;  // (i+1,3)
                            for (GB_HASH (i))       // find i in hash table
                            {
                                int64_t hf = Hf [hash] ;
                                int64_t h = (hf >> 2) ;
                                if (hf == i_phase2)
                                { 
                                    // C(i,j) has been seen; update it.
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    GB_HX_UPDATE (hash, t) ;// Hx [hash] += t
                                    break ;
                                }
                                else if (hf == i_phase1)
                                { 
                                    // first time C(i,j) seen here
                                    Hf [hash] = i_phase2 ;
                                    GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                    GB_HX_WRITE (hash, t) ; // Hx [hash] = t
                                    Ci [pC++] = i ;
                                    break ;
                                }
                                else if (h == i1) break ; // ignore i
                                ASSERT (h != 0) ;
                            }
                        }
                    }
                }

                //--------------------------------------------------------------
                // sort and gather C(:,j) for all 1-vector coarse hash tasks
                //--------------------------------------------------------------

                // found i if: Hf [hash] == (((i+1) << 2) + 3)
                GB_SORT_AND_GATHER_HASHED_C_j (((i+1) << 2) + 3, true) ;

            }
            #endif
            else
            {

                //--------------------------------------------------------------
                // phase3: multi-vector coarse hash task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;
                int64_t hash_bits = (hash_size-1) ;

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase3: multi-vector coarse hash task, C=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let f = Hf [hash] and h = Hi [hash]

                    // f < mark          : unoccupied.
                    // h == i, f == mark : occupied with C(i,j)

//                  printf ("%3d phase3: #vec coarse hash C=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;
                        int64_t cjnz = Cp [kk+1] - pC ;
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_B_j ;                // get B(:,j)
                        if (bjnz == 1)              // C(:,j) = A(:,k)*B(k,j)
                        { 
                            GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE ;
                            continue ;
                        }
                        mark++ ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
                            GB_GET_B_kj ;               // bkj = B(k,j)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k)
                                GB_MULT_A_ik_B_kj ;     // t = A(i,k)*B(k,j)
                                for (GB_HASH (i))   // find i in hash table
                                {
                                    if (Hf [hash] == mark)
                                    {
                                        // hash entry is occupied
                                        if (Hi [hash] == i)
                                        { 
                                            // i already in the hash table
                                            // Hx [hash] += t ;
                                            GB_HX_UPDATE (hash, t) ;
                                            break ;
                                        }
                                    }
                                    else
                                    { 
                                        // hash entry is not occupied
                                        Hf [hash] = mark ;
                                        Hi [hash] = i ;
                                        GB_HX_WRITE (hash, t) ;// Hx[hash]=t
                                        Ci [pC++] = i ;
                                        break ;
                                    }
                                }
                            }
                        }
                        // found i if: Hf [hash] == mark and Hi [hash] == i
                        GB_SORT_AND_GATHER_HASHED_C_j (mark, Hi [hash] == i)
                    }

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase3: multi-vector coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark            : M(i,j)=0, C(i,j) is ignored.
                    // h == i, f == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // h == i, f == mark+1 : M(i,j)=1, and C(i,j) has been seen.

//                  printf ("%3d phase3: #vec coarse hash C<M>=A*B\n", taskid) ;
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;
                        int64_t cjnz = Cp [kk+1] - pC ;
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_M_j ;                // get M(:,j)
                        GB_GET_M_j_RANGE (64) ;     // get 1st & last in M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;               // hash M(:,j)
                        GB_GET_B_j ;                // get B(:,j)
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            GB_SKIP_IF_A_k_DISJOINT_WITH_M_j ;
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
                    // phase3: multi-vector coarse hash task, C<!M>=A*B
                    //----------------------------------------------------------

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark: unoccupied, M(i,j)=0, and C(i,j) not yet seen.
                    // h == i, f == mark   : M(i,j)=1. C(i,j) ignored.
                    // h == i, f == mark+1 : M(i,j)=0, and C(i,j) has been seen.

//                  printf ("%3d phase3: #vec coarse hash C<!M>=A*B\n", taskid);
                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        int64_t pC = Cp [kk] ;
                        int64_t cjnz = Cp [kk+1] - pC ;
                        if (cjnz == 0) continue ;   // nothing to do
                        GB_GET_M_j ;                // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;               // hash M(:,j)
                        GB_GET_B_j ;                // get B(:,j)
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
                            GB_GET_B_kj ;               // bkj = B(k,j)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k)
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
                                        Ci [pC++] = i ;
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

    tic = omp_get_wtime ( ) - tic ;
    printf ("phase3: %g\n", tic) ;
    tic = omp_get_wtime ( ) ;

    //==========================================================================
    // phase4: gather phase for fine tasks
    //==========================================================================

//  printf ("phase4\n") ;

    // cumulative sum of nnz (C (:,j)) for each team of fine tasks
    int64_t cjnz_sum = 0 ;
    int64_t cjnz_max = 0 ;
    for (taskid = 0 ; taskid < nfine ; taskid++)
    {
        if (taskid == TaskList [taskid].master)
        {
            cjnz_sum = 0 ;
            // also find the max (C (:,j)) for any fine hash tasks
            int64_t hash_size = TaskList [taskid].hsize ;
            bool use_Gustavson = (hash_size == cvlen) ;
            if (!use_Gustavson)
            { 
                int64_t kk = TaskList [taskid].vector ;
                int64_t cjnz = Cp [kk+1] - Cp [kk] ;
                cjnz_max = GB_IMAX (cjnz_max, cjnz) ;
            }
        }
        int64_t my_cjnz = TaskList [taskid].my_cjnz ;
        TaskList [taskid].my_cjnz = cjnz_sum ;
        cjnz_sum += my_cjnz ;
    }

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kk = TaskList [taskid].vector ;
        #if !GB_IS_ANY_PAIR_SEMIRING
        GB_CTYPE *GB_RESTRICT Hx = TaskList [taskid].Hx ;
        #endif
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        int64_t pC = Cp [kk] ;
        int64_t cjnz = Cp [kk+1] - pC ;
        pC += TaskList [taskid].my_cjnz ;
        int team_size = TaskList [taskid].team_size ;
        int master = TaskList [taskid].master ;
        int my_teamid = taskid - master ;

        //----------------------------------------------------------------------
        // gather the values into C(:,j)
        //----------------------------------------------------------------------

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // phase4: fine Gustavson task, C=A*B, C<M>=A*B, or C<!M>=A*B
            //------------------------------------------------------------------

            // Hf [i] == 2 if C(i,j) is an entry in C(:,j)

            // printf ("%3d phase4: fine Gustavson\n", taskid) ;
            uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            if (cjnz < cvlen)   // work done in phase3 if cjnz == cvlen
            {

                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cvlen, my_teamid, team_size) ;

                if (M == NULL)
                {

                    for (int64_t i = istart ; i < iend ; i++)
                    {
                        if (Hf [i] == 2)
                        { 
                            #if !GB_IS_ANY_PAIR_SEMIRING
                            GB_CIJ_GATHER (pC, i) ; // Cx [pC] = Hx [i]
                            #endif
                            Ci [pC++] = i ;
                        }
                    }

                }
                else if (mask_is_M)
                {
                    // TODO

                }
                else
                {

#if 0
                    bool mjdense = (mjnz == cvlen) ;
                    if (mjdense)
                    {
printf (".") ;
                        uint32_t *GB_RESTRICT Mz = (uint32_t *) Mx ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
                            GB_GET_B_kj ;               // bkj = B(k,j)
                            // scan A(:,k), and lookup M(i,j)
                            for ( ; pA < pA_end ; pA++)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k) 
                                // do C(i,j)<M(i,j)> += A(i,k) * B(k,j) for this method
                                // M(i,j) may be 0 or 1, as given in the hash table
                                // GB_GET_M_ij (pM + i) ;     // get M(i,j)
                                if (!Mz [pM + i])
                                Hf [i] = 2 ;      // Hf [i] = M(i,j)
                            }
                        }
                    }
                    else
#endif

                    GB_GET_M_j ;                // get M(:,j)
#if 0
                    bool mjdense = (mjnz == cvlen) ;
                    if (mjdense)
                    {
// printf ("-") ;
                        uint32_t *GB_RESTRICT Mz = (uint32_t *) Mx ;
                        for (int64_t i = istart ; i < iend ; i++)
                        {

// GB_GET_M_ij (pM + i) ;     // get M(i,j)
// if (mij != (Mz [pM + i] != 0)) { printf ("#") ; abort ( ) ; }

                            // if (!Mz [pM + i] && Hf [i] == 2)
                            // if (Hf [i] == 2)

                            if (!Mz [pM + i] && Hf [i] == 2)
                            { 
                                Ci [pC++] = i ;
                            }
                        }
                    }
                    else
#endif
                    {
                        for (int64_t i = istart ; i < iend ; i++)
                        {
                            if (Hf [i] == 2)
                            { 
                                Ci [pC++] = i ;
                            }
                        }
                    }

                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // phase4: fine hash task, C=A*B, C<M>=A*B, C<!M>=A*B
            //------------------------------------------------------------------

            // (Hf [hash] & 3) == 2 if C(i,j) is an entry in C(:,j),
            // and the index i of the entry is (Hf [hash] >> 2) - 1.

//          printf ("%3d phase4: fine hash\n", taskid) ;
            int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            int64_t mystart, myend ;
            GB_PARTITION (mystart, myend, hash_size, my_teamid, team_size) ; 
            for (int64_t hash = mystart ; hash < myend ; hash++)
            {
                int64_t hf = Hf [hash] ;
                if ((hf & 3) == 2)
                { 
                    int64_t i = (hf >> 2) - 1 ; // found C(i,j) in hash table
                    Ci [pC++] = i ;
                }
            }
        }
    }

    tic = omp_get_wtime ( ) - tic ;
    printf ("phase4: %g\n", tic) ;
    tic = omp_get_wtime ( ) ;

    //==========================================================================
    // phase5: final gather phase for fine hash tasks
    //==========================================================================

//  printf ("phase5\n") ;

    if (cjnz_max > 0)
    {
        int64_t *GB_RESTRICT W = NULL ;
        bool parallel_sort = (cjnz_max > GB_BASECASE && nthreads > 1) ;
        if (parallel_sort)
        {
            // allocate workspace for parallel mergesort
            GB_MALLOC_MEMORY (W, cjnz_max, sizeof (int64_t)) ;
            if (W == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GB_OUT_OF_MEMORY) ;
            }
        }

        for (taskid = 0 ; taskid < nfine ; taskid++)
        {
            int64_t hash_size = TaskList [taskid].hsize ;
            bool use_Gustavson = (hash_size == cvlen) ;
            if (!use_Gustavson && taskid == TaskList [taskid].master)
            {

                //--------------------------------------------------------------
                // phase5: fine hash task, C=A*B, C<M>=A*B, C<!M>=A*B
                //--------------------------------------------------------------

                // (Hf [hash] & 3) == 2 if C(i,j) is an entry in C(:,j),
                // and the index i of the entry is (Hf [hash] >> 2) - 1.

                int64_t kk = TaskList [taskid].vector ;
                int64_t hash_bits = (hash_size-1) ;
                int64_t  *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                int64_t cjnz = Cp [kk+1] - Cp [kk] ;

                // sort the pattern of C(:,j)
                int nth = GB_nthreads (cjnz, chunk, nthreads) ;
                if (parallel_sort && nth > 1)
                { 
                    // parallel mergesort
                    GB_msort_1 (Ci + Cp [kk], W, cjnz, nth) ;
                }
                else
                { 
                    // sequential quicksort
                    GB_qsort_1a (Ci + Cp [kk], cjnz) ;
                }

                #if !GB_IS_ANY_PAIR_SEMIRING

                    GB_CTYPE *GB_RESTRICT Hx = TaskList [taskid].Hx ;
                    // gather the values of C(:,j)
                    int64_t pC ;
                    #pragma omp parallel for num_threads(nth) schedule(static)
                    for (pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)
                    {
                        int64_t i = Ci [pC] ;   // get C(i,j)
                        int64_t i1 = i + 1 ;
                        for (GB_HASH (i))       // find i in hash table
                        {
                            int64_t hf = Hf [hash] ;
                            if ((hf & 3) == 2 && (hf >> 2) == i1)
                            { 
                                // found i in the hash table
                                GB_CIJ_GATHER (pC, hash) ; // Cx[pC] = Hx[hash]
                                break ;
                            }
                        }
                    }

                #endif
            }
        }

        // free workspace
        GB_FREE_MEMORY (W, cjnz_max, sizeof (int64_t)) ;
    }

    tic = omp_get_wtime ( ) - tic ;
    printf ("phase5: %g\n", tic) ;

//  printf ("phases done\n") ;
}

#undef GB_GET_M_j
#undef GB_GET_M_j_RANGE
#undef GB_SCATTER_M_j
#undef GB_HASH_M_j
#undef GB_GET_B_j
#undef GB_GET_B_kj
#undef GB_GET_A_k
#undef GB_SKIP_IF_A_k_DISJOINT_WITH_M_j
#undef GB_GET_M_ij
#undef GB_MULT_A_ik_B_kj
#undef GB_COMPUTE_DENSE_C_j
#undef GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE
#undef GB_GATHER_ALL_C_j
#undef GB_SORT_AND_GATHER_C_j
#undef GB_SORT_AND_GATHER_HASHED_C_j
#undef GB_ATOMIC_UPDATE
#undef GB_ATOMIC_WRITE
#undef GB_HASH
#undef GB_FREE_ALL
#undef t
#undef Cx
#undef Hx

    }
    return (GrB_SUCCESS) ;
    #endif
}

#endif

