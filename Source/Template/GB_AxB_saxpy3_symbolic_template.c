//------------------------------------------------------------------------------
// GB_AxB_saxpy3_symbolic_template: symbolic analysis for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------
// phase1: symbolic analysis
//------------------------------------------------------------------------------

// fine tasks: the mask is scatter into the hash table, unless the mask is
// dense and accessed in-place.  No work to do in this phase if there is no
// mask.

// coarse tasks: computed nnz (C (:,j)) for each vector in the task.

{
    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
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

            // get M(:,j) and partition it
            GB_GET_M_j ;
            int team_size = TaskList [taskid].team_size ;
            int master    = TaskList [taskid].master ;
            int my_teamid = taskid - master ;
            int64_t mystart, myend ;
            GB_PARTITION (mystart, myend, mjnz, my_teamid, team_size) ;
            mystart += pM_start ;
            myend   += pM_start ;

            if (use_Gustavson)
            { 

                //--------------------------------------------------------------
                // phase1: fine Gustavson task, C<M>=A*B or C<!M>=A*B
                //--------------------------------------------------------------

                // Scatter the values of M(:,j) into H [...].f.  No atomics
                // needed since all indices i in M(;,j) are unique.  Do not
                // scatter the mask if M(:,j) is a dense vector, since in that
                // case the numeric phase accesses M(:,j) directly, not via
                // H [...].f.

                // M_dense_in_place is true only if M is dense, and all tasks
                // are fine or coarse hash tasks (no Gustvason tasks).
                // The mask M may be dense or full, sparse, or hypersparse.
                ASSERT (!M_dense_in_place) ;

                if (mjnz > 0)
                {
                    GB_HASH_FINEGUS *GB_RESTRICT H =
                        (GB_HASH_FINEGUS *) TaskList [master].H ;
                    ASSERT (H != NULL) ;
                    GB_SCATTER_M_j (mystart, myend, 1) ;
                }

            }
            else if (!M_dense_in_place)
            {

                //--------------------------------------------------------------
                // phase1: fine hash task, C<M>=A*B or C<!M>=A*B
                //--------------------------------------------------------------

                // If M_dense_in_place is true, this is skipped.  The mask M
                // is dense, and is used in place.

                // The least significant 2 bits of H [hash].f is the flag f,
                // and the upper bits contain h, as (h,f).  After this phase1,
                // if M(i,j)=1 then the hash table contains ((i+1),1) in H
                // [hash].f at some location.

                // Later, the flag values of f = 2 and 3 are also used.
                // Only f=1 is set in this phase.

                // h == 0,   f == 0: unoccupied and unlocked
                // h == i+1, f == 1: occupied with M(i,j)=1

                int64_t hash_bits = (hash_size-1) ;
                GB_HASH_TYPE *GB_RESTRICT H = 
                    (GB_HASH_TYPE *) TaskList [master].H ;
                ASSERT (H != NULL) ;
                // scan my M(:,j)
                for (int64_t pM = mystart ; pM < myend ; pM++)
                {
                    GB_GET_M_ij ;               // get M(i,j)
                    if (!mij) continue ;        // skip if M(i,j)=0
                    int64_t i = GBI (Mi, pM, mvlen) ;
                    int64_t i_mine = ((i+1) << 2) + 1 ; // ((i+1),1)
                    for (GB_HASH (i))
                    { 
                        int64_t hf ;
                        // swap my hash entry into the hash table;
                        // does the following using an atomic capture:
                        // { hf = H [hash].f ; H [hash].f = i_mine ; }
                        GB_ATOMIC_CAPTURE_INT64 (hf, H [hash].f, i_mine) ;
                        if (hf == 0) break ;        // success
                        // i_mine has been inserted, but a prior entry was
                        // already there.  It needs to be replaced, so take
                        // ownership of this displaced entry, and keep looking
                        // until a new empty slot is found for it.
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

            int64_t kfirst = TaskList [taskid].start ;
            int64_t klast  = TaskList [taskid].end ;
            int64_t mark = 0 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase1: coarse Gustavson task
                //--------------------------------------------------------------

                GB_HASH_TYPE *GB_RESTRICT H =
                    (GB_HASH_TYPE *GB_RESTRICT) TaskList [taskid].H ;
                ASSERT (H != NULL) ;

                if (M == NULL)
                {

                    //----------------------------------------------------------
                    // phase1: coarse Gustavson task, C=A*B
                    //----------------------------------------------------------

                    // Initially, H [...].f < mark for all H [...].f.
                    // H [i].f is set to mark when C(i,j) is found.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0)
                        { 
                            Cp [kk] = 0 ;       // ok: C is sparse
                            continue ;
                        }
                        if (bjnz == 1)
                        { 
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;            // get A(:,k)
                            Cp [kk] = aknz ;        // nnz(C(:,j)) = nnz(A(:,k))
                            continue ;
                        }
                        mark++ ;
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == cvlen)
                            { 
                                cjnz = cvlen ;  // A(:,k) is dense
                                break ;         // so nnz(C(:,j)) = cvlen
                            }
                            // scan A(:,k)
                            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                            {
                                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)
                                if (H [i].f != mark)     // if true, i is new
                                { 
                                    H [i].f = mark ; // mark C(i,j) as seen
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

                    // Initially, H [...].f < mark for all of H [...].f.

                    // H [i].f < mark    : M(i,j)=0, C(i,j) is ignored.
                    // H [i].f == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // H [i].f == mark+1 : M(i,j)=1, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0)
                        { 
                            Cp [kk] = 0 ;       // ok: C is sparse
                            continue ;
                        }
                        GB_GET_M_j ;            // get M(:,j)
                        if (mjnz == 0)
                        { 
                            Cp [kk] = 0 ;       // ok: C is sparse
                            continue ;
                        }
                        GB_GET_M_j_RANGE (64) ;
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_SCATTER_M_j (pM_start, pM_end, mark) ; // scatter Mj
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        { 
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
                            #define GB_IKJ                                     \
                            {                                                  \
                                if (H [i].f == mark)  /* if true, M(i,j) is 1*/\
                                {                                              \
                                    H [i].f = mark1 ;  /* mark C(i,j) seen */  \
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

                    // Initially, H [...].f < mark for all of H [...].f.

                    // H [i].f < mark    : M(i,j)=0, C(i,j) is not yet seen.
                    // H [i].f == mark   : M(i,j)=1, so C(i,j) is ignored.
                    // H [i].f == mark+1 : M(i,j)=0, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0)
                        { 
                            Cp [kk] = 0 ;       // ok: C is sparse
                            continue ;
                        }
                        GB_GET_M_j ;            // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_SCATTER_M_j (pM_start, pM_end, mark) ; // scatter Mj
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            // scan A(:,k)
                            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                            {
                                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)
                                if (H [i].f < mark)      // if true, M(i,j) is 0
                                { 
                                    H [i].f = mark1 ;    // mark C(i,j) as seen
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

                GB_HASH_COARSE *GB_RESTRICT H =
                    (GB_HASH_COARSE *GB_RESTRICT) TaskList [taskid].H ;
                int64_t hash_bits = (hash_size-1) ;
                ASSERT (H != NULL) ;

                if (M == NULL || ignore_mask)
                {

                    //----------------------------------------------------------
                    // phase1: coarse hash task, C=A*B
                    //----------------------------------------------------------

                    // no mask present, or the mask is C<M>=A*B, dense, not
                    // complemented, and structural (and thus effectively no
                    // mask at all).
                    #undef GB_CHECK_MASK_ij
                    #include "GB_AxB_saxpy3_coarseHash_phase1.c"

                }
                else if (mask_is_M)
                {

                    //----------------------------------------------------------
                    // phase1: coarse hash task, C<M>=A*B
                    //----------------------------------------------------------

                    if (M_dense_in_place)
                    { 
                        // M(:,j) is dense.  M is not scattered into H [...].f.
                        ASSERT (Mx != NULL) ; // ignore_mask case handled above
                        #define GB_CHECK_MASK_ij if (Mask [i] == 0) continue ;
                        switch (msize)
                        {
                            default:
                            case 1:
                            {
                                #define M_TYPE uint8_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 2:
                            {
                                #define M_TYPE uint16_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 4:
                            {
                                #define M_TYPE uint32_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 8:
                            {
                                #define M_TYPE uint64_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 16:
                            {
                                #define M_TYPE uint64_t
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                      \
                                    if (Mask [2*i] == 0 && Mask [2*i+1] == 0) \
                                        continue ;
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                        }
                    }

                    // Initially, H [...].f < mark for all of H [...].f
                    // Let h = H [hash].i and f = H [hash].f.

                    // f < mark: unoccupied, M(i,j)=0, C(i,j) ignored if
                    //           this case occurs while scanning A(:,k)
                    // h == i, f == mark   : M(i,j)=1, and C(i,j) not yet seen.
                    // h == i, f == mark+1 : M(i,j)=1, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0)
                        { 
                            Cp [kk] = 0 ;       // ok: C is sparse
                            continue ;
                        }
                        GB_GET_M_j ;            // get M(:,j)
                        if (mjnz == 0)
                        { 
                            Cp [kk] = 0 ;       // ok: C is sparse
                            continue ;
                        }
                        GB_GET_M_j_RANGE (64) ;
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;           // hash M(:,j)
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        { 
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            if (aknz == 0) continue ;
                            #define GB_IKJ                                     \
                            {                                                  \
                                for (GB_HASH (i))       /* find i in hash */   \
                                {                                              \
                                    int64_t f = H [hash].f ;                   \
                                    if (f < mark) break ; /* M(i,j)=0; ignore*/\
                                    if (H [hash].i == i)  /* if true, i found*/\
                                    {                                          \
                                        if (f == mark)  /* if true, i is new */\
                                        {                                      \
                                            H [hash].f = mark1 ; /* seen */    \
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
                        // M(:,j) is dense.  M is not scattered into H [...].f.
                        if (Mx == NULL)
                        { 
                            // structural mask, complemented.  No work to do.
                            #ifdef GB_DEBUG
                            for (int64_t kk = kfirst ; kk <= klast ; kk++)
                            { 
                                ASSERT (Cp [kk] == 0) ;
                            }
                            #endif
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
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 2:
                            {
                                #define M_TYPE uint16_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 4:
                            {
                                #define M_TYPE uint32_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 8:
                            {
                                #define M_TYPE uint64_t
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                            case 16:
                            {
                                #define M_TYPE uint64_t
                                #define M_SIZE 2
                                #undef  GB_CHECK_MASK_ij
                                #define GB_CHECK_MASK_ij                      \
                                    if (Mask [2*i] != 0 || Mask [2*i+1] != 0) \
                                        continue ;
                                #include "GB_AxB_saxpy3_coarseHash_phase1.c"
                            }
                        }
                    }

                    // Initially, H [...].f < mark for all of H [...].f.
                    // Let h = H [hash].i and f = H [hash].f

                    // f < mark: unoccupied, M(i,j)=0, and C(i,j) not yet seen.
                    // h == i, f == mark   : M(i,j)=1. C(i,j) ignored.
                    // h == i, f == mark+1 : M(i,j)=0, and C(i,j) has been seen.

                    for (int64_t kk = kfirst ; kk <= klast ; kk++)
                    {
                        GB_GET_B_j ;            // get B(:,j)
                        if (bjnz == 0)
                        { 
                            Cp [kk] = 0 ;       // ok: C is sparse
                            continue ;
                        }
                        GB_GET_M_j ;            // get M(:,j)
                        mark += 2 ;
                        int64_t mark1 = mark+1 ;
                        GB_HASH_M_j ;           // hash M(:,j)
                        int64_t cjnz = 0 ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = GBI (Bi, pB, bvlen) ;   // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            // scan A(:,k)
                            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                            {
                                int64_t i = GBI (Ai, pA, avlen) ; // get A(i,k)
                                for (GB_HASH (i))       // find i in hash
                                {
                                    if (H [hash].f < mark)  // if true, i is new
                                    { 
                                        H [hash].f = mark1 ; // mark C(i,j) seen
                                        H [hash].i = i ;
                                        cjnz++ ;        // C(i,j) is a new entry
                                        break ;
                                    }
                                    if (H [hash].i == i) break ;
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
            int master = TaskList [taskid].master ;
            if (master != taskid) continue ;
            GB_GET_M_j ;        // get M(:,j)
            if (mjnz == 0) continue ;
            int64_t mjcount2 = 0 ;
            int64_t mjcount = 0 ;
            for (int64_t pM = pM_start ; pM < pM_end ; pM++)
            {
                GB_GET_M_ij ;           // get M(i,j)
                if (mij) mjcount++ ;
            }
            if (use_Gustavson)
            {
                // phase1: fine Gustavson task, C<M>=A*B or C<!M>=A*B
                GB_HASH_FINEGUS *GB_RESTRICT H =
                    (GB_HASH_FINEGUS *GB_RESTRICT) TaskList [master].H ;
                ASSERT (H != NULL) ;
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    GB_GET_M_ij ;                    // get M(i,j)
                    ASSERT (H [GBI (Mi, pM, mvlen)].f == mij) ;
                }
                for (int64_t i = 0 ; i < cvlen ; i++)
                {
                    ASSERT (H [i].f == 0 || H [i].f == 1) ;
                    if (H [i].f == 1) mjcount2++ ;
                }
                ASSERT (mjcount == mjcount2) ;
            }
            else if (!M_dense_in_place)
            {
                // phase1: fine hash task, C<M>=A*B or C<!M>=A*B
                // h == 0,   f == 0: unoccupied and unlocked
                // h == i+1, f == 1: occupied with M(i,j)=1
                GB_HASH_TYPE *GB_RESTRICT H =
                    (GB_HASH_TYPE *GB_RESTRICT) TaskList [master].H ;
                int64_t hash_bits = (hash_size-1) ;
                ASSERT (H != NULL) ;
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    GB_GET_M_ij ;                   // get M(i,j)
                    if (!mij) continue ;            // skip if M(i,j)=0
                    int64_t i = GBI (Mi, pM, mvlen) ;
                    int64_t i_mine = ((i+1) << 2) + 1 ;  // ((i+1),1)
                    int64_t probe = 0 ;
                    for (GB_HASH (i))
                    {
                        int64_t hf = H [hash].f ;
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
                    int64_t hf = H [hash].f ;
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
break ;

#undef GB_HASH_FINEGUS
#undef GB_HASH_TYPE
#undef GB_HASH_COARSE

