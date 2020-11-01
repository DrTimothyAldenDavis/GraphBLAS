//------------------------------------------------------------------------------
// GB_AxB_saxpy3_symbolic: symbolic analysis for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Symbolic analysis for C=A*B, C<M>=A*B or C<!M>=A*B, via GB_AxB_saxpy3.
// Coarse tasks compute nnz (C (:,j)) for each of their vectors j.  Fine tasks
// just scatter the mask M into the hash table.  This phase does not depend on
// the semiring, nor does it depend on the type of C, A, or B.  It does access
// the values of M, if the mask matrix M is present and not structural.

// If B is hypersparse, C must also be hypersparse.
// Otherwise, C must be sparse.

#include "GB_AxB_saxpy3.h"
#include "GB_AxB_saxpy3_template.h"
#include "GB_atomics.h"
#include "GB_bracket.h"
// GB_GET_A_k and GB_GET_M_j declare aknz and mjnz, but these are unused here.
#include "GB_unused.h"

void GB_AxB_saxpy3_symbolic
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_comp,       // M complemented, or not
    const bool Mask_struct,     // M structural, or not
    const bool M_dense_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *TaskList,     // list of tasks, and workspace
    int ntasks,                 // total number of tasks
    int nfine,                  // number of fine tasks
    int nthreads                // number of threads
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_PENDING (M)) ; 

    ASSERT (!GB_ZOMBIES (A)) ; 
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ; 

    ASSERT (!GB_ZOMBIES (B)) ; 
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_PENDING (B)) ; 

    //--------------------------------------------------------------------------
    // get M, A, B, and C
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Cp = C->p ;
    const int64_t cvlen = C->vlen ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int8_t  *GB_RESTRICT Bb = B->b ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const int64_t bvlen = B->vlen ;
    const bool B_jumbled = B->jumbled ;
    const bool B_is_sparse_or_hyper = GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B);
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int8_t  *GB_RESTRICT Ab = A->b ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const bool A_is_hyper = GB_IS_HYPER (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_jumbled = A->jumbled ;

    const int64_t *GB_RESTRICT Mp = NULL ;
    const int64_t *GB_RESTRICT Mh = NULL ;
    const int8_t  *GB_RESTRICT Mb = NULL ;
    const int64_t *GB_RESTRICT Mi = NULL ;
    const GB_void *GB_RESTRICT Mx = NULL ;
    size_t msize = 0 ;
    int64_t mnvec = 0 ;
    int64_t mvlen = 0 ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    if (M != NULL)
    {   GB_cov[2165]++ ;
// covered (2165): 40488
        Mp = M->p ;
        Mh = M->h ;
        Mb = M->b ;
        Mi = M->i ;
        Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
        mnvec = M->nvec ;
        mvlen = M->vlen ;
    }

    // 3 cases:
    //      M not present and Mask_comp false: compute C=A*B
    //      M present     and Mask_comp false: compute C<M>=A*B
    //      M present     and Mask_comp true : compute C<!M>=A*B
    // If M is NULL on input, then Mask_comp is also false on input.

    const bool mask_is_M = (M != NULL && !Mask_comp) ;

    // ignore the mask if present, not complemented, dense and
    // used in place, structural, and not bitmap.  In this case,
    // all entries in M are true, so M can be ignored.
    const bool ignore_mask = mask_is_M && M_dense_in_place &&
        Mask_struct && !M_is_bitmap ;

    //==========================================================================
    // phase1: count nnz(C(:,j)) for coarse tasks, scatter M for fine tasks
    //==========================================================================

    // At this point, all of Hf [...] is zero, for all tasks.
    // Hi and Hx are not initialized.

    int taskid ;
// TODO #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;

        if (taskid < nfine)
        {

            //------------------------------------------------------------------
            // no work for fine tasks in phase1 if M is not present
            //------------------------------------------------------------------

            if (M == NULL) continue ;

            //------------------------------------------------------------------
            // get the task descriptor
            //------------------------------------------------------------------

            int64_t kk = TaskList [taskid].vector ;
            int64_t bjnz = (Bp == NULL) ? bvlen : (Bp [kk+1] - Bp [kk]) ;
            // no work to do if B(:,j) is empty
            if (bjnz == 0) continue ;

            // partition M(:,j)
            GB_GET_M_j ;        // get M(:,j)

            int team_size = TaskList [taskid].team_size ;
            int leader    = TaskList [taskid].leader ;
            int my_teamid = taskid - leader ;
            int64_t mystart, myend ;
            GB_PARTITION (mystart, myend, mjnz, my_teamid, team_size) ;
            mystart += pM_start ;
            myend   += pM_start ;

            if (use_Gustavson)
            {   GB_cov[2166]++ ;
// covered (2166): 246833

                //--------------------------------------------------------------
                // phase1: fine Gustavson task, C<M>=A*B or C<!M>=A*B
                //--------------------------------------------------------------

                // Scatter the values of M(:,j) into Hf.  No atomics needed
                // since all indices i in M(;,j) are unique.  Do not scatter
                // the mask if M(:,j) is a dense vector, since in that case
                // the numeric phase accesses M(:,j) directly, not via Hf.

                // M_dense_in_place is true only if M is dense, and all tasks
                // are fine or coarse hash tasks (no Gustvason tasks).
                ASSERT (!M_dense_in_place) ;

                if (mjnz > 0)
                {   GB_cov[2167]++ ;
// covered (2167): 246822
                    int8_t *GB_RESTRICT
                        Hf = (int8_t *GB_RESTRICT) TaskList [taskid].Hf ;
                    GB_SCATTER_M_j (mystart, myend, 1) ;
                }

            }
            else if (!M_dense_in_place)
            {

                //--------------------------------------------------------------
                // phase1: fine hash task, C<M>=A*B or C<!M>=A*B
                //--------------------------------------------------------------

                // If M_dense_in_place is true, this is skipped.  The mask M
                // is dense, and is used in-place.

                // The least significant 2 bits of Hf [hash] is the flag f, and
                // the upper bits contain h, as (h,f).  After this phase1, if
                // M(i,j)=1 then the hash table contains ((i+1),1) in Hf [hash]
                // at some location.

                // Later, the flag values of f = 2 and 3 are also used.
                // Only f=1 is set in this phase.

                // h == 0,   f == 0: unoccupied and unlocked
                // h == i+1, f == 1: occupied with M(i,j)=1

                int64_t *GB_RESTRICT
                    Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
                int64_t hash_bits = (hash_size-1) ;
                // scan my M(:,j)
                for (int64_t pM = mystart ; pM < myend ; pM++)
                {
                    GB_GET_M_ij (pM) ;              // get M(i,j)
                    if (!mij) continue ;            // skip if M(i,j)=0
                    int64_t i = GBI (Mi, pM, mvlen) ;
                    int64_t i_mine = ((i+1) << 2) + 1 ;  // ((i+1),1)
                    for (GB_HASH (i))
                    {   GB_cov[2168]++ ;
// covered (2168): 1713
                        int64_t hf ;
                        // swap my hash entry into the hash table;
                        // does the following using an atomic capture:
                        // { hf = Hf [hash] ; Hf [hash] = i_mine ; }
                        GB_ATOMIC_CAPTURE_INT64 (hf, Hf [hash], i_mine) ;
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
        else
        {

            //------------------------------------------------------------------
            // coarse tasks: compute nnz in each vector of A*B(:,kfirst:klast)
            //------------------------------------------------------------------

            int64_t *GB_RESTRICT
                Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
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

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        Cp [kk] = 0 ;           // ok: C is sparse
                        if (bjnz == 0) continue ;
                        if (bjnz == 1 && !A_is_bitmap)
                        {   GB_cov[2169]++ ;
// covered (2169): 54287
                            if (!GBB (Bb, pB)) continue ;
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;            // get A(:,k)
                            Cp [kk] = aknz ;        // nnz(C(:,j)) = nnz(A(:,k))
                            continue ;
                        }
                        mark++ ;
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            if (!GBB (Bb, pB)) continue ;
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == cvlen && !A_is_bitmap)
                            {   GB_cov[2170]++ ;
// covered (2170): 510828
                                cjnz = cvlen ;  // A(:,k) is dense
                                break ;         // so nnz(C(:,j)) = cvlen
                            }
                            // scan A(:,k)
                            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                            {
                                if (!GBB (Ab, pA)) continue ;
                                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)
                                if (Hf [i] != mark)     // if true, i is new
                                {   GB_cov[2171]++ ;
// covered (2171): 50390269
                                    Hf [i] = mark ; // mark C(i,j) as seen
                                    cjnz++ ;        // C(i,j) is a new entry
                                }
                            }
                        }
                        // count the entries in C(:,j)
                        Cp [kk] = cjnz ;            // ok: C is sparse
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

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        Cp [kk] = 0 ;           // ok: C is sparse
                        if (bjnz == 0) continue ;
                        GB_GET_M_j ;            // get M(:,j)
                        if (mjnz == 0) continue ;
                        GB_GET_M_j_RANGE (64) ;
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_SCATTER_M_j (pM_start, pM_end, mark) ; // scatter Mj
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {   GB_cov[2172]++ ;
// covered (2172): 12291901
                            if (!GBB (Bb, pB)) continue ;
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
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
                        // count the entries in C(:,j)
                        Cp [kk] = cjnz ;        // ok: C is sparse
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

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        Cp [kk] = 0 ;           // ok: C is sparse
                        if (bjnz == 0) continue ;
                        GB_GET_M_j ;            // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_SCATTER_M_j (pM_start, pM_end, mark) ; // scatter Mj
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            if (!GBB (Bb, pB)) continue ;
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            // scan A(:,k)
                            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                            {
                                if (!GBB (Ab, pA)) continue ;
                                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)
                                if (Hf [i] < mark)      // if true, M(i,j) is 0
                                {   GB_cov[2173]++ ;
// covered (2173): 59213
                                    Hf [i] = mark1 ;    // mark C(i,j) as seen
                                    cjnz++ ;            // C(i,j) is a new entry
                                }
                            }
                        }
                        // count the entries in C(:,j)
                        Cp [kk] = cjnz ;        // ok: C is sparse
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase1: coarse hash task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;
                int64_t hash_bits = (hash_size-1) ;

                if (M == NULL || ignore_mask)
                {   GB_cov[2174]++ ;
// covered (2174): 9591

                    //----------------------------------------------------------
                    // phase1: coarse hash task, C=A*B
                    //----------------------------------------------------------

                    // no mask present, or mask ignored
                    #undef GB_CHECK_MASK_ij
                    #include "GB_AxB_saxpy3_coarseHash_phase1.c"

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase1: coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    if (M_dense_in_place)
                    {   GB_cov[2175]++ ;
// covered (2175): 7

                        // M(:,j) is dense.  M is not scattered into Hf.

                        ASSERT (!Mask_struct || M_is_bitmap) ;
                        #define GB_CHECK_MASK_ij                        \
                            bool mij =                                  \
                                (M_is_bitmap ? Mjb [i] : 1) &&          \
                                (Mask_struct ? 1 : (Mjx [i] != 0)) ;    \
                            if (!mij) continue ;

                        switch (msize)
                        {
                            default:
                            case 1  : GB_cov[2176]++ ;  
// covered (2176): 7
                            {
                                #define M_TYPE uint8_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 2  : GB_cov[2177]++ ;  
// NOT COVERED (2177):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint16_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 4  : GB_cov[2178]++ ;  
// NOT COVERED (2178):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint32_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 8  : GB_cov[2179]++ ;  
// NOT COVERED (2179):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint64_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 16  : GB_cov[2180]++ ;  
// NOT COVERED (2180):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint64_t
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                    \
                                    bool mij =                              \
                                        (M_is_bitmap ? Mjb [i] : 1) &&      \
                                        (Mask_struct ? 1 :                  \
                                            (Mjx [2*i] != 0) ||             \
                                            (Mjx [2*i+1] != 0)) ;           \
                                    if (!mij) continue ;

                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                        }
                    }

                    // Initially, Hf [...] < mark for all of Hf.
                    // Let h = Hi [hash] and f = Hf [hash].

                    // f < mark: unoccupied, M(i,j)=0, C(i,j) ignored if
                    //           this case occurs while scanning A(:,k)
                    // h == i, f == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // h == i, f == mark+1 : M(i,j)=1, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        Cp [kk] = 0 ;           // ok: C is sparse
                        if (bjnz == 0) continue ;
                        GB_GET_M_j ;            // get M(:,j)
                        if (mjnz == 0) continue ;
                        GB_GET_M_j_RANGE (64) ;
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;           // hash M(:,j)
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {   GB_cov[2181]++ ;
// covered (2181): 28575
                            if (!GBB (Bb, pB)) continue ;
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
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
                        // count the entries in C(:,j)
                        Cp [kk] = cjnz ;        // ok: C is sparse
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // phase1: coarse hash task, C<!M>=A*B
                    //----------------------------------------------------------

                    if (M_dense_in_place)
                    {

                        // M(:,j) is dense.  M is not scattered into Hf.

                        if (Mask_struct && !M_is_bitmap)
                        {   GB_cov[2182]++ ;
// NOT COVERED (2182):
GB_GOTCHA ;
                            // structural mask, complemented, not bitmap.
                            // No work to do.
                            #ifdef GB_DEBUG
                            for (int64_t kk = kfirst ; kk <= klast ; kk++)
                            {
                                ASSERT (Cp [kk] == 0) ;
                            }
                            #endif
                            continue ;
                        }

                        #undef  GB_CHECK_MASK_ij
                        #define GB_CHECK_MASK_ij                        \
                            bool mij =                                  \
                                (M_is_bitmap ? Mjb [i] : 1) &&          \
                                (Mask_struct ? 1 : (Mjx [i] != 0)) ;    \
                            if (mij) continue ;

                        switch (msize)
                        {
                            default:
                            case 1  : GB_cov[2183]++ ;  
// covered (2183): 5
                            {
                                #define M_TYPE uint8_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 2  : GB_cov[2184]++ ;  
// NOT COVERED (2184):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint16_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 4  : GB_cov[2185]++ ;  
// NOT COVERED (2185):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint32_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 8  : GB_cov[2186]++ ;  
// NOT COVERED (2186):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint64_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 16  : GB_cov[2187]++ ;  
// NOT COVERED (2187):
GB_GOTCHA ;
                            {
                                #define M_TYPE uint64_t
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                    \
                                    bool mij =                              \
                                        (M_is_bitmap ? Mjb [i] : 1) &&      \
                                        (Mask_struct ? 1 :                  \
                                            (Mjx [2*i] != 0) ||             \
                                            (Mjx [2*i+1] != 0)) ;           \
                                    if (mij) continue ;

                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
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
                        GB_GET_B_j ;            // get B(:,j)
                        Cp [kk] = 0 ;           // ok: C is sparse
                        if (bjnz == 0) continue ;
                        GB_GET_M_j ;            // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;           // hash M(:,j)
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            if (!GBB (Bb, pB)) continue ;
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            // scan A(:,k)
                            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                            {
                                if (!GBB (Ab, pA)) continue ;
                                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)
                                for (GB_HASH (i))       // find i in hash
                                {
                                    if (Hf [hash] < mark)   // if true, i is new
                                    {   GB_cov[2188]++ ;
// covered (2188): 100401
                                        Hf [hash] = mark1 ; // mark C(i,j) seen
                                        Hi [hash] = i ;
                                        cjnz++ ;        // C(i,j) is a new entry
                                        break ;
                                    }
                                    if (Hi [hash] == i) break ;
                                }
                            }
                        }
                        // count the entries in C(:,j)
                        Cp [kk] = cjnz ;        // ok: C is sparse
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // check result for phase1 for fine tasks
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    if (M != NULL)
    {
        for (taskid = 0 ; taskid < nfine ; taskid++)
        {
            int64_t kk = TaskList [taskid].vector ;
            ASSERT (kk >= 0 && kk < B->nvec) ;
            int64_t bjnz = (Bp == NULL) ? bvlen : (Bp [kk+1] - Bp [kk]) ;
            // no work to do if B(:,j) is empty
            if (bjnz == 0) continue ;
            int64_t hash_size = TaskList [taskid].hsize ;
            bool use_Gustavson = (hash_size == cvlen) ;
            int leader = TaskList [taskid].leader ;
            if (leader != taskid) continue ;
            GB_GET_M_j ;        // get M(:,j)
            if (mjnz == 0) continue ;
            int64_t mjcount2 = 0 ;
            int64_t mjcount = 0 ;
            for (int64_t pM = pM_start ; pM < pM_end ; pM++)
            {
                GB_GET_M_ij (pM) ;                  // get M(i,j)
                if (mij) mjcount++ ;
            }
            if (use_Gustavson)
            {
                // phase1: fine Gustavson task, C<M>=A*B or C<!M>=A*B
                int8_t *GB_RESTRICT
                    Hf = (int8_t *GB_RESTRICT) TaskList [taskid].Hf ;
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    GB_GET_M_ij (pM) ;               // get M(i,j)
                    int64_t i = GBI (Mi, pM, mvlen) ;
                    ASSERT (Hf [i] == mij) ;
                }
                for (int64_t i = 0 ; i < cvlen ; i++)
                {
                    ASSERT (Hf [i] == 0 || Hf [i] == 1) ;
                    if (Hf [i] == 1) mjcount2++ ;
                }
                ASSERT (mjcount == mjcount2) ;
            }
            else if (!M_dense_in_place)
            {
                // phase1: fine hash task, C<M>=A*B or C<!M>=A*B
                // h == 0,   f == 0: unoccupied and unlocked
                // h == i+1, f == 1: occupied with M(i,j)=1
                int64_t *GB_RESTRICT
                    Hf = (int64_t *GB_RESTRICT) TaskList [taskid].Hf ;
                int64_t hash_bits = (hash_size-1) ;
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    GB_GET_M_ij (pM) ;              // get M(i,j)
                    if (!mij) continue ;            // skip if M(i,j)=0
                    int64_t i = GBI (Mi, pM, mvlen) ;
                    int64_t i_mine = ((i+1) << 2) + 1 ;  // ((i+1),1)
                    int64_t probe = 0 ;
                    for (GB_HASH (i))
                    {
                        int64_t hf = Hf [hash] ;
                        if (hf == i_mine) 
                        {
                            mjcount2++ ;
                            break ;
                        }
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
    }
    #endif
}

