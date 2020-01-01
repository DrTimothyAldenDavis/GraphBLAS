//------------------------------------------------------------------------------
// GB_AxB_saxpy3_template: C=A*B via saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_AxB_saxpy3_template.c computes C=A*B for any semiring and matrix types.
// It is #include'd in GB_AxB_saxpy3 to construct the generic method (for
// arbitary run-time defined operators and/or typecasting), in the hard-coded
// GB_Asaxpy3B* workers in the Generated/ folder, and in the
// compile-time-constructed functions called by GB_AxB_user.

//------------------------------------------------------------------------------
// macros
//------------------------------------------------------------------------------

// prepare to iterate over the vector B(:,j), the (kk)th vector in B,
// where j == ((Bh == NULL) ? kk : Bh [kk]).  Note that j itself is never
// needed; just kk.
#define GB_GET_B_j                                                          \
    int64_t pleft = 0 ;                                                     \
    int64_t pright = anvec-1 ;                                              \
    int64_t pB = Bp [kk] ;                                                  \
    int64_t pB_end = Bp [kk+1] ;                                            \
    int64_t bjnz = pB_end - pB ;                                            \
    if (A_is_hyper && bjnz > 2)                                             \
    {                                                                       \
        /* trim Ah [0..pright] to remove any entries past last B(:,j), */   \
        /* to speed up GB_lookup in GB_GET_A_k.  Note that this assumes */  \
        /* the indices in B(:,j) are sorted; otherwise the entire */        \
        /* computation could be done with B having jumbled vectors. */      \
        GB_bracket_right (Bi [pB_end-1], Ah, 0, &pright) ;                  \
    }

// prepare to iterate over the vector A(:,k)
#define GB_GET_A_k                                                      \
    int64_t pA, pA_end ;                                                \
    GB_lookup (A_is_hyper, Ah, Ap, &pleft, pright, k, &pA, &pA_end)     \

// ctype t = A(i,k) * B(k,j)
#define GB_T_EQ_AIK_TIMES_BKJ                               \
    GB_GETA (aik, Ax, pA) ;     /* aik = Ax [pA] ;  */      \
    GB_CIJ_DECLARE (t) ;        /* ctype t ;        */      \
    GB_MULT (t, aik, bkj)       /* t = aik * bkj ;  */

// atomic update
#if GB_HAS_ATOMIC
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
            while (!__sync_bool_compare_and_swap                    \
                ((GB_CTYPE_PUN *) px,                               \
                * ((GB_CTYPE_PUN *) (&xold)),                       \
                * ((GB_CTYPE_PUN *) (&xnew))))
    #endif
#else
    // Hx [i] += t via critical section
    #define GB_ATOMIC_UPDATE(i,t)       \
        GB_PRAGMA (omp flush)           \
        GB_HX_UPDATE (i, t) ;           \
        GB_PRAGMA (omp flush)
#endif

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

// hash functions
#define GB_HASH_FUNCTION(i) ((i * GB_HASH_FACTOR) & (hash_bits))
#define GB_REHASH(hash) hash = ((hash + 1) & (hash_bits))

// to iterate over the hash table, looking for index i:
// for (GB_HASH (i)) { ... }
#define GB_HASH(i) int64_t hash = GB_HASH_FUNCTION (i) ; ; GB_REHASH (hash)

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

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Cp = C->p ;
    // const int64_t *GB_RESTRICT Ch = C->h ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnvec = C->nvec ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    // const int64_t *GB_RESTRICT Bh = B->h ;
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

    //==========================================================================
    // phase1: count nnz(C(:,j)) for all tasks; do numeric work for fine tasks
    //==========================================================================

    // Coarse tasks: compute nnz (C(:,kfirst:klast)).
    // Fine tasks: compute nnz (C(:,j)), and values in Hx via atomics.

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kk = TaskList [taskid].vector ;
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        bool is_fine = (kk >= 0) ;

        if (is_fine)
        {

            //------------------------------------------------------------------
            // fine task: compute nnz (C(:,j)) and values in Hx
            //------------------------------------------------------------------

            int64_t pB     = TaskList [taskid].start ;
            int64_t pB_end = TaskList [taskid].end + 1 ;
            int64_t my_cjnz = 0 ;
            GB_CTYPE *GB_RESTRICT Hx = TaskList [taskid].Hx ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase1: fine Gustavson task
                //--------------------------------------------------------------

                // All of Hf [...] is initially zero.
                // Hf [i] == 0: unlocked, i has not been seen in C(:,j).
                //              Hx [i] is not initialized.
                // Hf [i] == 1: unlocked, i has been seen in C(:,j).
                //              Hx [i] is initialized.
                // Hf [i] == 2: locked.  Hx [i] cannot be accessed.

                uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                int64_t pleft = 0, pright = anvec-1 ;
                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                {
                    int64_t k = Bi [pB] ;       // get B(k,j)
                    GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                    GB_GET_A_k ;                // get A(:,k)
                    for ( ; pA < pA_end ; pA++) // scan A(:,k)
                    {
                        int64_t i = Ai [pA] ;       // get A(i,k)
                        GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k) * B(k,j)
                        int64_t f ;
                        #if GB_HAS_ATOMIC
                        // grab the entry from the hash table
                        #pragma omp atomic read
                        f = Hf [i] ;
                        if (f == 1)
                        { 
                            // C(i,j) is already initialized; update it
                            GB_ATOMIC_UPDATE (i, t) ;   // Hx [i] += t
                        }
                        else
                        #endif
                        {
                            // lock the entry
                            do
                            {
                                #pragma omp atomic capture
                                {
                                    f = Hf [i] ; Hf [i] = 2 ;
                                }
                            } while (f == 2) ;
                            // the owner of the lock gets f == 0 if this is
                            // the first time the entry has been seen, or
                            // f == 1 if this is the 2nd time seen.  If f==2
                            // is returned, the lock has not been obtained.
                            if (f == 0)
                            { 
                                // C(i,j) is a new entry in C(:,j)
                                GB_ATOMIC_WRITE (i, t) ;    // Hx [i] = t
                            }
                            else // f == 1
                            { 
                                // C(i,j) already appears in C(:,j)
                                GB_ATOMIC_UPDATE (i, t) ;   // Hx [i] += t
                            }
                            // unlock the entry
                            #pragma omp atomic write
                            Hf [i] = 1 ;
                            if (f == 0)
                            { 
                                my_cjnz++ ; // C(i,j) is a new entry
                            }
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
                // is the least significant bit.  h is 63 bits, and is the
                // 1-based index i of the C(i,j) entry stored at that location
                // in the hash table.  All of Hf [...] is initially zero.

                // Given Hf [hash] split into (h,f):
                // h == 0, f == 0: unlocked, hash entry is unoccupied.
                //                  Hx [hash] is not initialized.
                // h == i+1, f == 0: unlocked, hash entry contains C(i,j).
                //                  Hx [hash] is initialized. This case (i+1,0)
                //                  is equal to i_unlocked.
                // h == 0, f == 1: locked, hash entry is unoccupied.
                //                  Hx [hash] is not initialized.
                // h == i+1, f == 1: locked, hash entry contains C(i,j).
                //                  Hx [hash] is initialized.

                int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                int64_t hash_bits = (hash_size-1) ;
                int64_t pleft = 0, pright = anvec-1 ;
                for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                {
                    int64_t k = Bi [pB] ;       // get B(k,j)
                    GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                    GB_GET_A_k ;                // get A(:,k)
                    for ( ; pA < pA_end ; pA++) // scan A(:,k)
                    {
                        int64_t i = Ai [pA] ;       // get A(i,k)
                        GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k) * B(k,j)
                        int64_t i1 = i + 1 ;        // i1 = one-based index
                        int64_t i_unlocked = (i1 << 1) ;        // (i+1,0)
                        #ifdef GB_DEBUG
                        int64_t probe = 0 ;
                        #endif
                        for (GB_HASH (i))           // find i in hash table
                        {
                            int64_t hf ;
                            // grab the entry from the hash table
                            #pragma omp atomic read
                            hf = Hf [hash] ;
                            #if GB_HAS_ATOMIC
                            if (hf == i_unlocked)
                            {
                                // C(i,j) is already initialized; update it.
                                GB_ATOMIC_UPDATE (hash, t) ;// Hx [hash] += t
                                break ;
                            }
                            #endif
                            int64_t h = (hf >> 1) ;
                            if (h == 0 || h == i1)
                            {
                                // h == 0: unoccupied
                                // h == i1: occupied by C(i,j)
                                // lock the entry
                                do
                                {
                                    #pragma omp atomic capture
                                    {
                                        hf = Hf [hash] ; Hf [hash] |= 1 ;
                                    }
                                } while (hf & 1) ;
                                if (hf == 0)
                                { 
                                    // C(i,j) is a new entry in C(:,j)
                                    GB_ATOMIC_WRITE (hash, t) ; // Hx [hash] = t
                                    // unlock the entry
                                    #pragma omp atomic write
                                    Hf [hash] = i_unlocked ;
                                    my_cjnz++ ;
                                    break ;
                                }
                                else if (hf == i_unlocked)
                                { 
                                    // C(i,j) already appears in C(:,j)
                                    GB_ATOMIC_UPDATE (hash, t) ;// Hx[hash] += t
                                    // unlock the entry
                                    #pragma omp atomic write
                                    Hf [hash] = i_unlocked ;
                                    break ;
                                }
                                else
                                { 
                                    // hash table occupied, but not with C(i,j)
                                    // unlock the entry
                                    #pragma omp atomic write
                                    Hf [hash] = hf ;
                                }
                            }
                            #ifdef GB_DEBUG
                            probe++ ;
                            ASSERT (probe < cvlen) ;
                            #endif
                        }
                    }
                }
            }

            TaskList [taskid].my_cjnz = my_cjnz ;

        }
        else
        {

            //------------------------------------------------------------------
            // coarse task: compute nnz in each vector of A*B(:,kfirst:klast)
            //------------------------------------------------------------------

            // TODO make this a single function used by all semirings

            int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            int64_t kfirst = TaskList [taskid].start ;
            int64_t klast  = TaskList [taskid].end ;
            int64_t mark = 0 ;
            int64_t nk = klast - kfirst + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase1: coarse Gustavson task
                //--------------------------------------------------------------

                for (int64_t kk = kfirst ; kk <= klast ; kk++)
                {
                    // count the entries in C(:,j)
                    // where j == ((Bh == NULL) ? kk : Bh [kk])
                    int64_t cjnz = 0 ;
                    GB_GET_B_j ;            // get B(:,j)
                    if (bjnz == 1)
                    { 
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        cjnz = pA_end - pA ;        // # nonzeros in C(:,j)
                    }
                    else if (bjnz > 1)
                    {
                        mark++ ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            // TODO handle the case when A(:,k) is dense
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;       // get A(i,k)
                                if (Hf [i] != mark)
                                { 
                                    Hf [i] = mark ;     // C(i,j) is a new entry
                                    cjnz++ ;
                                }
                            }
                        }
                    }
                    Cp [kk] = cjnz ;
                }

            }
            else if (nk == 1)
            {

                //--------------------------------------------------------------
                // phase1: 1-vector coarse hash task
                //--------------------------------------------------------------

                // Hf [hash] is zero if the hash entry is empty, or
                // ((i+1) << 1) if it contains entry i.

                int64_t hash_bits = (hash_size-1) ;
                int64_t kk = kfirst ;
                int64_t cjnz = 0 ;
                GB_GET_B_j ;            // get B(:,j)
                if (bjnz == 1)
                { 
                    int64_t k = Bi [pB] ;           // get B(k,j)
                    GB_GET_A_k ;                    // get A(:,k)
                    cjnz = pA_end - pA ;            // # nonzeros in C(:,j)
                }
                else if (bjnz > 1)
                {
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            int64_t i1 = (i+1) << 1 ;
                            #ifdef GB_DEBUG
                            int64_t probe = 0 ;
                            #endif
                            for (GB_HASH (i))           // find i in hash table
                            {
                                int64_t h = Hf [hash] ;
                                if (h == i1)
                                { 
                                    break ; // already in hash table
                                }
                                else if (h == 0)
                                { 
                                    Hf [hash] = i1 ; // C(i,j) is a new entry
                                    cjnz++ ;
                                    break ;
                                }
                                #ifdef GB_DEBUG
                                probe++ ;
                                ASSERT (probe < cvlen) ;
                                #endif
                            }
                        }
                    }
                }
                Cp [kk] = cjnz ;

            }
            else
            {

                //--------------------------------------------------------------
                // phase1: multi-vector coarse hash task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;
                int64_t hash_bits = (hash_size-1) ;
                for (int64_t kk = kfirst ; kk <= klast ; kk++)
                {
                    // count the entries in C(:,j)
                    int64_t cjnz = 0 ;
                    GB_GET_B_j ;            // get B(:,j)
                    if (bjnz == 1)
                    { 
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GET_A_k ;                // get A(:,k)
                        cjnz = pA_end - pA ;        // # nonzeros in C(:,j)
                    }
                    else if (bjnz > 1)
                    {
                        mark++ ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GET_A_k ;                // get A(:,k)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k)
                                #ifdef GB_DEBUG
                                int64_t probe = 0 ;
                                #endif
                                for (GB_HASH (i))       // find i in hash table
                                {
                                    if (Hf [hash] == mark)
                                    {
                                        // hash entry is occupied
                                        if (Hi [hash] == i)
                                        { 
                                            break ; // already in hash table
                                        }
                                    }
                                    else
                                    { 
                                        // C(i,j) is a new entry.
                                        Hf [hash] = mark ;
                                        Hi [hash] = i ;
                                        cjnz++ ;
                                        break ;
                                    }
                                    #ifdef GB_DEBUG
                                    probe++ ;
                                    ASSERT (probe < cvlen) ;
                                    #endif
                                }
                            }
                        }
                    }
                    Cp [kk] = cjnz ;
                }
            }
        }
    }

    //==========================================================================
    // phase2: compute Cp with cumulative sum and allocate C
    //==========================================================================

    // TODO make this a single function used by all semirings

    // TaskList [taskid].my_cjnz is the # of unique entries found in C(:,j) by
    // that task.  Sum these terms to compute total # of entries in C(:,j).

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    { 
        int64_t kk = TaskList [taskid].vector ;
        Cp [kk] = 0 ;
    }

    for (int taskid = 0 ; taskid < nfine ; taskid++)
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

    //==========================================================================
    // phase3: numeric phase for coarse tasks, prep for gather for fine tasks
    //==========================================================================

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        GB_CTYPE *GB_RESTRICT Hx = TaskList [taskid].Hx ;
        int64_t kk = TaskList [taskid].vector ;
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        bool is_fine = (kk >= 0) ;

        if (is_fine)
        {
            // TODO make this a single function used by all semirings
            // Note the memcpy from Hx to Cx

            //------------------------------------------------------------------
            // count nnz (C(:,j) for the final gather for this fine task
            //------------------------------------------------------------------

            int nfine_team_size = TaskList [taskid].nfine_team_size ;
            int master    = TaskList [taskid].master ;
            int my_teamid = taskid - master ;
            int64_t my_cjnz = 0 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase3: fine Gustavson task
                //--------------------------------------------------------------

                int64_t pC = Cp [kk] ;
                int64_t cjnz = Cp [kk+1] - pC ;
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cvlen, my_teamid, nfine_team_size) ;
                if (cjnz == cvlen)
                {
                    // C(:,j) is entirely dense: finish the work now
                    for (int64_t i = istart ; i < iend ; i++)
                    { 
                        Ci [pC + i] = i ;
                    }
                    // copy Hx [istart:iend-1] into Cx [pC+istart:pC+iend-1]
                    GB_CIJ_MEMCPY (pC + istart, istart, iend - istart) ;
                }
                else
                {
                    // C(:,j) is sparse: count the work for this fine task
                    uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                    // O(cvlen) linear scan of Hf to create the pattern of
                    // C(:,j).  No sort is needed.
                    for (int64_t i = istart ; i < iend ; i++)
                    {
                        if (Hf [i])
                        { 
                            my_cjnz++ ;
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase3: fine hash task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                int64_t hash_start, hash_end ;
                GB_PARTITION (hash_start, hash_end, hash_size,
                    my_teamid, nfine_team_size) ;
                for (int64_t hash = hash_start ; hash < hash_end ; hash++)
                {
                    if (Hf [hash] != 0)
                    { 
                        my_cjnz++ ;
                    }
                }
            }

            TaskList [taskid].my_cjnz = my_cjnz ;

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
            int64_t mark = nk + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // phase3: coarse Gustavson task
                //--------------------------------------------------------------

                for (int64_t kk = kfirst ; kk <= klast ; kk++)
                {
                    // compute the pattern and values of C(:,j)
                    GB_GET_B_j ;            // get B(:,j)
                    if (bjnz == 0)
                    { 
                        continue ;          // nothing to do
                    }
                    int64_t pC = Cp [kk] ;
                    int64_t cjnz = Cp [kk+1] - pC ;
                    if (bjnz == 1)          // C(:,j) = A(:,k)*B(k,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                        GB_GET_A_k ;                // get A(:,k)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        { 
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k) * B(k,j)
                            GB_CIJ_WRITE (pC, t) ;      // Cx [pC] = t
                            Ci [pC++] = i ;
                        }
                    }
                    else if (cjnz == cvlen)     // C(:,j) is dense
                    {
                        for (int64_t i = 0 ; i < cvlen ; i++)
                        { 
                            Ci [pC + i] = i ;
                            GB_CIJ_WRITE (pC + i, GB_IDENTITY) ; // C(i,j) = 0
                        }
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                            GB_GET_A_k ;                // get A(:,k)
                            // TODO handle the case when A(:,k) is dense
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            { 
                                int64_t i = Ai [pA] ;       // get A(i,k)
                                GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k)*B(k,j)
                                GB_CIJ_UPDATE (pC + i, t) ; // Cx [pC+i] += t
                            }
                        }
                    }
                    else if (16 * cjnz > cvlen) // C(:,j) is not very sparse
                    {
                        mark++ ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                            GB_GET_A_k ;                // get A(:,k)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;       // get A(i,k)
                                GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k)*B(k,j)
                                if (Hf [i] != mark)
                                { 
                                    // C(i,j) = A(i,k) * B(k,j)
                                    Hf [i] = mark ;
                                    GB_HX_WRITE (i, t) ;    // Hx [i] = t ;
                                }
                                else
                                { 
                                    // C(i,j) += A(i,k) * B(k,j)
                                    GB_HX_UPDATE (i, t) ;   // Hx [i] += t ;
                                }
                            }
                        }
                        // gather the pattern and values into C(:,j)
                        for (int64_t i = 0 ; i < cvlen ; i++)
                        {
                            if (Hf [i] == mark)
                            { 
                                GB_CIJ_GATHER (pC, i) ;   // Cx [pC] = Hx [i] ;
                                Ci [pC++] = i ;
                            }
                        }
                    }
                    else if (cjnz > 0)  // C(:,j) is very sparse
                    {
                        mark++ ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                            GB_GET_A_k ;                // get A(:,k)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;       // get A(i,k)
                                GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k)*B(k,j)
                                if (Hf [i] != mark)
                                { 
                                    // C(i,j) = A(i,k) * B(k,j)
                                    Hf [i] = mark ;
                                    GB_HX_WRITE (i, t) ;    // Hx [i] = t ;
                                    Ci [pC++] = i ;
                                }
                                else
                                { 
                                    // C(i,j) += A(i,k) * B(k,j)
                                    GB_HX_UPDATE (i, t) ;   // Hx [i] += t ;
                                }
                            }
                        }
                        // sort the pattern of C(:,j)
                        GB_qsort_1a (Ci + Cp [kk], cjnz) ;
                        // gather the values into C(:,j)
                        for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)
                        { 
                            int64_t i = Ci [pC] ;
                            GB_CIJ_GATHER (pC, i) ;   // Cx [pC] = Hx [i] ;
                        }
                    }
                }

            }
            else if (nk == 1)
            {

                //--------------------------------------------------------------
                // phase3: 1-vector coarse hash task
                //--------------------------------------------------------------

                int64_t hash_bits = (hash_size-1) ;
                int64_t kk = kfirst ;
                GB_GET_B_j ;            // get B(:,j)
                if (bjnz == 0)
                { 
                    continue ;          // nothing to do
                }
                int64_t pC = Cp [kk] ;
                int64_t cjnz = Cp [kk+1] - pC ;
                if (bjnz == 1)          // C(:,j) = A(:,k)*B(k,j)
                {
                    int64_t k = Bi [pB] ;       // get B(k,j)
                    GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                    GB_GET_A_k ;                // get A(:,k)
                    for ( ; pA < pA_end ; pA++) // scan A(:,k)
                    { 
                        int64_t i = Ai [pA] ;       // get A(i,k)
                        GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k) * B(k,j)
                        GB_CIJ_WRITE (pC, t) ;      // Cx [pC] = t
                        Ci [pC++] = i ;
                    }
                }
                else
                {
                    // Hf [hash] has been set to ((i+1)<<1) in phase1, for all
                    // entries i in the pattern of C(:,j).  The first time
                    // Hf [hash] is seen here in phase3, it is incremented to
                    // ((i+1)<<1)+1 to denote it has been seen here in phase3.
                    for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                    {
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                        GB_GET_A_k ;                // get A(:,k)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        {
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            int64_t i1 = (i+1) << 1 ;
                            int64_t i2 = i1 + 1 ;
                            GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k)*B(k,j)
                            #ifdef GB_DEBUG
                            int64_t probe = 0 ;
                            #endif
                            for (GB_HASH (i))           // find i in hash table
                            {
                                int64_t h = Hf [hash] ;
                                if (h == i2)
                                { 
                                    // C(i,j) has been seen before; update it.
                                    GB_HX_UPDATE (hash, t) ; // Hx [hash] += t ;
                                    break ;
                                }
                                else if (h == i1)
                                { 
                                    // first time C(i,j) seen here
                                    Hf [hash] = i2 ;
                                    GB_HX_WRITE (hash, t) ; // Hx [hash] = t ;
                                    Ci [pC++] = i ;
                                    break ;
                                }
                                #ifdef GB_DEBUG
                                probe++ ;
                                ASSERT (probe < cvlen) ;
                                #endif
                            }
                        }
                    }
                    // sort the pattern of C(:,j)
                    GB_qsort_1a (Ci + Cp [kk], cjnz) ;
                    // gather the values of C(:,j)
                    for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)
                    {
                        int64_t i = Ci [pC] ;
                        int64_t i2 = ((i+1) << 1) + 1 ;
                        #ifdef GB_DEBUG
                        int64_t probe = 0 ;
                        #endif
                        for (GB_HASH (i))           // find i in hash table
                        {
                            if (Hf [hash] == i2)
                            { 
                                // i found in the hash table
                                GB_CIJ_GATHER (pC, hash) ;  // Cx[pC] = Hx[hash]
                                break ;
                            }
                            #ifdef GB_DEBUG
                            probe++ ;
                            ASSERT (probe < cvlen) ;
                            #endif
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // phase3: multi-vector coarse task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;
                int64_t hash_bits = (hash_size-1) ;
                for (int64_t kk = kfirst ; kk <= klast ; kk++)
                {
                    // compute the pattern and values of C(:,j)
                    GB_GET_B_j ;            // get B(:,j)
                    if (bjnz == 0)
                    { 
                        continue ;          // nothing to do
                    }
                    int64_t pC = Cp [kk] ;
                    int64_t cjnz = Cp [kk+1] - pC ;
                    if (bjnz == 1)          // C(:,j) = A(:,k)*B(k,j)
                    {
                        // C(:,j) = A(:,k)*B(k,j) for a single entry B(k,j)
                        int64_t k = Bi [pB] ;       // get B(k,j)
                        GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                        GB_GET_A_k ;                // get A(:,k)
                        for ( ; pA < pA_end ; pA++) // scan A(:,k)
                        { 
                            int64_t i = Ai [pA] ;       // get A(i,k)
                            GB_T_EQ_AIK_TIMES_BKJ ;     // t = A(i,k) * B(k,j)
                            GB_CIJ_WRITE (pC, t) ;      // Cx [pC] = t
                            Ci [pC++] = i ;
                        }
                    }
                    else
                    {
                        mark++ ;
                        for ( ; pB < pB_end ; pB++)     // scan B(:,j)
                        {
                            int64_t k = Bi [pB] ;       // get B(k,j)
                            GB_GETB (bkj, Bx, pB) ;     // bkj = Bx [pB] ;
                            GB_GET_A_k ;                // get A(:,k)
                            for ( ; pA < pA_end ; pA++) // scan A(:,k)
                            {
                                int64_t i = Ai [pA] ;   // get A(i,k)
                                GB_T_EQ_AIK_TIMES_BKJ ; // t = A(i,k)*B(k,j)
                                #ifdef GB_DEBUG
                                int64_t probe = 0 ;
                                #endif
                                for (GB_HASH (i))       // find i in hash table
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
                                        GB_HX_WRITE (hash, t) ; // Hx [hash] = t
                                        Ci [pC++] = i ;
                                        break ;
                                    }
                                    #ifdef GB_DEBUG
                                    probe++ ;
                                    ASSERT (probe < cvlen) ;
                                    #endif
                                }
                            }
                        }
                        // sort the pattern of C(:,j)
                        GB_qsort_1a (Ci + Cp [kk], cjnz) ;
                        // gather the values of C(:,j)
                        for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)
                        {
                            int64_t i = Ci [pC] ;
                            #ifdef GB_DEBUG
                            int64_t probe = 0 ;
                            #endif
                            for (GB_HASH (i))       // find i in hash table
                            {
                                if (Hi [hash] == i)
                                { 
                                    // i found in the hash table
                                    // Cx [pC] = Hx [hash] ;
                                    GB_CIJ_GATHER (pC, hash) ;
                                    break ;
                                }
                                #ifdef GB_DEBUG
                                probe++ ;
                                ASSERT (probe < cvlen) ;
                                #endif
                            }
                        }
                    }
                    ASSERT (pC == Cp [kk+1]) ;
                }
            }
        }
    }

    //==========================================================================
    // phase4: gather phase for fine tasks
    //==========================================================================

    // cumulative sum of nnz (C (:,j)) for each team of fine tasks
    int64_t cjnz_sum = 0 ;
    int64_t cjnz_max = 0 ;
    for (int taskid = 0 ; taskid < nfine ; taskid++)
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
        GB_CTYPE *GB_RESTRICT Hx = TaskList [taskid].Hx ;
        int64_t hash_size = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cvlen) ;
        int64_t pC = Cp [kk] ;
        int64_t cjnz = Cp [kk+1] - pC ;
        pC += TaskList [taskid].my_cjnz ;
        int nfine_team_size = TaskList [taskid].nfine_team_size ;
        int master = TaskList [taskid].master ;
        int my_teamid = taskid - master ;

        //----------------------------------------------------------------------
        // gather the values into C(:,j)
        //----------------------------------------------------------------------

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // phase4: fine Gustavson task
            //------------------------------------------------------------------

            uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            if (cjnz < cvlen)
            {
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cvlen, my_teamid, nfine_team_size);
                for (int64_t i = istart ; i < iend ; i++)
                {
                    if (Hf [i])
                    { 
                        GB_CIJ_GATHER (pC, i) ; // Cx [pC] = Hx [i]
                        Ci [pC++] = i ;
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // phase4: fine hash task
            //------------------------------------------------------------------

            int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            int64_t hash_start, hash_end ;
            GB_PARTITION (hash_start, hash_end, hash_size,
                my_teamid, nfine_team_size) ; 
            for (int64_t hash = hash_start ; hash < hash_end ; hash++)
            {
                int64_t hf = Hf [hash] ;
                if (hf != 0)
                { 
                    int64_t i = (hf >> 1) - 1 ; // found C(i,j) in hash table
                    Ci [pC++] = i ;
                }
            }
        }
    }

    //==========================================================================
    // phase5: final gather phase for fine hash tasks
    //==========================================================================

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

        for (int taskid = 0 ; taskid < nfine ; taskid++)
        {
            int64_t hash_size = TaskList [taskid].hsize ;
            bool use_Gustavson = (hash_size == cvlen) ;
            if (!use_Gustavson && taskid == TaskList [taskid].master)
            {

                //--------------------------------------------------------------
                // phase5: fine hash task
                //--------------------------------------------------------------

                int64_t kk = TaskList [taskid].vector ;
                int64_t hash_bits = (hash_size-1) ;
                int64_t  *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                GB_CTYPE *GB_RESTRICT Hx = TaskList [taskid].Hx ;
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

                // gather the values of C(:,j)
                int64_t pC ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)
                {
                    int64_t i = Ci [pC] ;   // get C(i,j)
                    int64_t i1 = i + 1 ;
                    #ifdef GB_DEBUG
                    int64_t probe = 0 ;
                    #endif
                    for (GB_HASH (i))       // find i in hash table
                    {
                        if ((Hf [hash] >> 1) == i1)
                        { 
                            // found i in the hash table
                            GB_CIJ_GATHER (pC, hash) ; // Cx [pC] = Hx [hash]
                            break ;
                        }
                        #ifdef GB_DEBUG
                        probe++ ;
                        ASSERT (probe < cvlen) ;
                        #endif
                    }
                }
            }
        }

        // free workspace
        GB_FREE_MEMORY (W, cjnz_max, sizeof (int64_t)) ;
    }
}

#undef GB_GET_B_j
#undef GB_GET_A_k
#undef GB_ATOMIC_UPDATE
#undef GB_ATOMIC_WRITE
#undef GB_HASH

