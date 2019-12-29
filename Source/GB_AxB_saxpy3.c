//------------------------------------------------------------------------------
// GB_AxB_saxpy3: compute C = A*B in parallel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This function only computes C=A*B.  The mask is not applied.

#define GB_DEBUG
#include "GB_AxB_saxpy3.h"

#define GB_FREE_WORK                                                    \
{                                                                       \
    GB_FREE_MEMORY (TaskList, max_ntasks+1, sizeof (GB_task_struct)) ;  \
}

#define GB_FREE_ALL                                                     \
{                                                                       \
    GB_FREE_WORK ;                                                      \
    GB_MATRIX_FREE (Chandle) ;                                          \
}

GrB_Info GB_AxB_saxpy3              // C = A*B using Gustavson+Hash
(
    GrB_Matrix *Chandle,            // output matrix
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    #if GB_TIMING
    printf ("\n---------- parallel version saxpy3\n") ;
    double tic = omp_get_wtime ( ) ;
    #endif

    GrB_Info info ;
    ASSERT (Chandle != NULL) ;
    ASSERT (*Chandle == NULL) ;
    ASSERT_MATRIX_OK (A, "A for saxpy3 A*B", GB0) ;
    ASSERT_MATRIX_OK (B, "B for saxpy3 A*B", GB0) ;
    ASSERT (!GB_PENDING (A)) ; ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (B)) ; ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT_SEMIRING_OK (semiring, "semiring for saxpy3 A*B", GB0) ;
    ASSERT (A->vdim == B->vlen) ;

    int64_t *GB_RESTRICT Hi_all = NULL ;
    int64_t *GB_RESTRICT Hf_all = NULL ;
    GB_void *GB_RESTRICT Hx_all = NULL ;
    int64_t *GB_RESTRICT Coarse_initial = NULL ;    // initial coarse tasks
    GB_hashtask_struct *GB_RESTRICT TaskList = NULL ;
    int64_t *GB_RESTRICT Ci = NULL ;
    GB_void *GB_RESTRICT Cx = NULL ;
    int64_t *GB_RESTRICT W = NULL ;
    int64_t *GB_RESTRICT Bflops2 = NULL ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;

    bool op_is_first  = mult->opcode == GB_FIRST_opcode ;
    bool op_is_second = mult->opcode == GB_SECOND_opcode ;
    bool A_is_pattern = false ;
    bool B_is_pattern = false ;

    if (flipxy)
    { 
        // z = fmult (b,a) will be computed
        A_is_pattern = op_is_first  ;
        B_is_pattern = op_is_second ;
        ASSERT (GB_IMPLIES (!A_is_pattern,
            GB_Type_compatible (A->type, mult->ytype))) ;
        ASSERT (GB_IMPLIES (!B_is_pattern,
            GB_Type_compatible (B->type, mult->xtype))) ;
    }
    else
    { 
        // z = fmult (a,b) will be computed
        A_is_pattern = op_is_second ;
        B_is_pattern = op_is_first  ;
        ASSERT (GB_IMPLIES (!A_is_pattern,
            GB_Type_compatible (A->type, mult->xtype))) ;
        ASSERT (GB_IMPLIES (!B_is_pattern,
            GB_Type_compatible (B->type, mult->ytype))) ;
    }

    (*Chandle) = NULL ;

    //--------------------------------------------------------------------------
    // get A, and B
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    // const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    // const int64_t anz = GB_NNZ (A) ;
    const int64_t anvec = A->nvec ;
    const bool A_is_hyper = A->is_hyper ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    // const int64_t *GB_RESTRICT Bi = B->i ;
    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    // const int64_t bnz = GB_NNZ (B) ;
    const int64_t bnvec = B->nvec ;
    const bool B_is_hyper = B->is_hyper ;

    //--------------------------------------------------------------------------
    // allocate C (but not C->i or C->x)
    //--------------------------------------------------------------------------

    GrB_Type ctype = add->op->ztype ;
    int64_t cvlen = avlen ;
    int64_t cvdim = bvdim ;
    int64_t cnz = 0 ;
    int64_t cnvec = bnvec ;
    bool C_is_hyper = (cvdim > 1) && (A_is_hyper || B_is_hyper) ;

    GB_NEW (Chandle, ctype, cvlen, cvdim, GB_Ap_malloc, true,
        GB_SAME_HYPER_AS (C_is_hyper), B->hyper_ratio, cnvec, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (info) ;
    }

    GrB_Matrix C = (*Chandle) ;

    int64_t *GB_RESTRICT Cp = C->p ;
    int64_t *GB_RESTRICT Ch = C->h ;
    size_t csize = ctype->size ;

    //--------------------------------------------------------------------------
    // determine the # of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //==========================================================================
    // phase0: create parallel tasks
    //==========================================================================

    //--------------------------------------------------------------------------
    // compute flop counts for each vector of B and C
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Bflops = Cp ;  // Cp is used as workspace for Bflops
    int64_t flmax = 1 ;
    int64_t total_flops = 0 ;
    bool flopresult ;

    GB_OK (GB_AxB_flopcount (&flopresult, Bflops, NULL, A, B, 0, Context)) ;
    int64_t total_flops = Bflops [bnvec] ;

    #if GB_TIMING
    double t1 = omp_get_wtime ( ) - tic ; ;
    tic = omp_get_wtime ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // determine # of threads and # of initial coarse tasks
    //--------------------------------------------------------------------------

    nthreads = GB_nthreads ((double) total_flops, chunk, nthreads_max) ;
    int ntasks_initial = (nthreads == 1) ?
        1 : (GB_NTASKS_PER_THREAD * nthreads) ;
    double target_task_size = ((double) total_flops) / ntasks_initial ;
    double target_fine_size = target_task_size / GB_FINE_WORK ;

    //--------------------------------------------------------------------------
    // determine # of parallel tasks
    //--------------------------------------------------------------------------

    int nfine = 0 ;         // # of fine tasks
    int ncoarse = 0 ;       // # of coarse tasks
    int64_t max_bjnz = 0 ;  // max (nnz (B (:,j))) of fine tasks

    if (ntasks_initial > 1)
    {

        //----------------------------------------------------------------------
        // construct initial coarse tasks
        //----------------------------------------------------------------------

        if (!GB_pslice (&Coarse_initial, Bflops, bnvec, ntasks_initial))
        {
            // out of memory
            GB_FREE_ALL ;
            return (GB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // split the work into coarse and fine tasks
        //----------------------------------------------------------------------

        for (int taskid = 0 ; taskid < ntasks_initial ; taskid++)
        {
            // get the initial coarse task
            int64_t j1 = Coarse_initial [taskid] ;
            int64_t j2 = Coarse_initial [taskid+1] ;
            int64_t task_ncols = j2 - j1 ;
            int64_t task_flops = Bflops [j2] - Bflops [j1] ;

            if (task_ncols == 0)
            {
                // This coarse task is empty, having been squeezed out by
                // costly vectors in adjacent coarse tasks.
            }
            else if (task_flops > 2 * GB_COSTLY * target_task_size)
            {
                // This coarse task is too costly, because it contains one or
                // more costly vectors.  Split its vectors into a mixture of
                // coarse and fine tasks.

                int64_t jcoarse_start = j1 ;

                for (int64_t j = j1 ; j < j2 ; j++)
                {
                    // jflops = # of flops to compute a single vector A*B(:,j)
                    double jflops = Bflops [j+1] - Bflops [j] ;
                    // bjnz = nnz (B (:,j))
                    int64_t bjnz = Bp [j+1] - Bp [j] ;

                    if (jflops > GB_COSTLY * target_task_size && bjnz > 1)
                    {
                        // A*B(:,j) is costly; split it into 2 or more fine
                        // tasks.  First flush the prior coarse task, if any.
                        if (jcoarse_start < j)
                        {
                            // columns jcoarse_start to j-1 form a single
                            // coarse task
                            ncoarse++ ;
                        }

                        // next coarse task (if any) starts at j+1
                        jcoarse_start = j+1 ;

                        // column j will be split into multiple fine tasks
                        max_bjnz = GB_IMAX (max_bjnz, bjnz) ;
                        int nfine_team_size = ceil (jflops / target_fine_size) ;
                        nfine += nfine_team_size ;
                    }
                }

                // flush the last coarse task, if any
                if (jcoarse_start < j2)
                {
                    // columns jcoarse_start to j2-1 form a single
                    // coarse task
                    ncoarse++ ;
                }

            }
            else
            {
                // This coarse task is OK as-is.
                ncoarse++ ;
            }
        }
    }
    else
    {

        //----------------------------------------------------------------------
        // entire computation in a single coarse task
        //----------------------------------------------------------------------

        nfine = 0 ;
        ncoarse = 1 ;
    }

    int ntasks = ncoarse + nfine ;

    //--------------------------------------------------------------------------
    // allocate the tasks, and workspace to construct fine tasks
    //--------------------------------------------------------------------------

    GB_CALLOC_MEMORY (TaskList, ntasks, sizeof (GB_hashtask_struct)) ;
    GB_MALLOC_MEMORY (W, ntasks+1, sizeof (int64_t)) ;
    if (nfine > 0)
    {
        GB_MALLOC_MEMORY (Bflops2, max_bjnz+1, sizeof (int64_t)) ;
    }

    if (TaskList == NULL || W == NULL || (nfine > 0 && Bflops2 == NULL))
    {
        // out of memory
        GB_FREE_ALL ;
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // create the tasks
    //--------------------------------------------------------------------------

    if (ntasks_initial > 1)
    {

        //----------------------------------------------------------------------
        // create the coarse and fine tasks
        //----------------------------------------------------------------------

        int nf = 0 ;        // fine tasks have task id 0:nfine-1
        int nc = nfine ;    // coarse task ids are nfine:ntasks-1

        for (int taskid = 0 ; taskid < ntasks_initial ; taskid++)
        {
            // get the initial coarse task
            int64_t j1 = Coarse_initial [taskid] ;
            int64_t j2 = Coarse_initial [taskid+1] ;
            int64_t task_ncols = j2 - j1 ;
            int64_t task_flops = Bflops [j2] - Bflops [j1] ;

            if (task_ncols == 0)
            {
                // This coarse task is empty, having been squeezed out by
                // costly vectors in adjacent coarse tasks.
            }
            else if (task_flops > 2 * GB_COSTLY * target_task_size)
            {
                // This coarse task is too costly, because it contains one or
                // more costly vectors.  Split its vectors into a mixture of
                // coarse and fine tasks.

                int64_t jcoarse_start = j1 ;

                for (int64_t j = j1 ; j < j2 ; j++)
                {
                    // jflops = # of flops to compute a single vector A*B(:,j)
                    double jflops = Bflops [j+1] - Bflops [j] ;
                    // bjnz = nnz (B (:,j))
                    int64_t bjnz = Bp [j+1] - Bp [j] ;

                    if (jflops > GB_COSTLY * target_task_size && bjnz > 1)
                    {
                        // A*B(:,j) is costly; split it into 2 or more fine
                        // tasks.  First flush the prior coarse task, if any.
                        if (jcoarse_start < j)
                        {
                            // jcoarse_start:j-1 form a single coarse task
                            GB_create_coarse_task (jcoarse_start, j-1,
                                TaskList, nc++, Bflops, cnrows,
                                chunk, nthreads_max) ;
                        }

                        // next coarse task (if any) starts at j+1
                        jcoarse_start = j+1 ;

                        // count the work for each B(k,j)
                        int64_t pB_start = Bp [j] ;
                        int nth = GB_nthreads (bjnz, chunk, nthreads_max) ;
                        #pragma omp parallel for num_threads(nth) \
                            schedule(static)
                        for (int64_t s = 0 ; s < bjnz ; s++)
                        {
                            // get B(k,j)
                            int64_t k = Bi [pB_start + s] ;
                            // flop count for just B(k,j)
                            int64_t fl = (Ap [k+1] - Ap [k]) ;
                            Bflops2 [s] = fl ;
                        }

                        // cumulative sum of flops to compute A*B(:,j)
                        GB_cumsum (Bflops2, bjnz, NULL, nth) ;

                        // slice B(:,j) into fine tasks
                        int nfine_team_size = ceil (jflops / target_fine_size) ;
                        GB_pslice (&W, Bflops2, bjnz, nfine_team_size) ;

                        // hash table for all fine takes for C(:,j)
                        int64_t hsize = GB_hash_table_size (jflops, cnrows) ;

                        // construct the fine tasks for B(:,j)
                        int master = nf ;
                        for (int fid = 0 ; fid < nfine_team_size ; fid++)
                        {
                            int64_t pstart = W [fid] ;
                            int64_t pend   = W [fid+1] ;
                            int64_t fl = Bflops2 [pend] - Bflops2 [pstart] ;
                            TaskList [nf].start  = pB_start + pstart ;
                            TaskList [nf].end    = pB_start + pend - 1 ;
                            TaskList [nf].vector = j ;
                            TaskList [nf].hsize  = hsize ;
                            TaskList [nf].master = master ;
                            TaskList [nf].nfine_team_size = nfine_team_size ;
                            TaskList [nf].flops = fl ;
                            nf++ ;
                        }
                    }
                }

                // flush the last coarse task, if any
                if (jcoarse_start < j2)
                {
                    // jcoarse_start:j-1 form a single coarse task
                    GB_create_coarse_task (jcoarse_start, j2-1,
                        TaskList, nc++, Bflops, cnrows, chunk, nthreads_max) ;
                }

            }
            else
            {
                // This coarse task is OK as-is.
                GB_create_coarse_task (j1, j2 - 1,
                    TaskList, nc++, Bflops, cnrows, chunk, nthreads_max) ;
            }
        }

        // free workspace
        GB_FREE_MEMORY (Bflops2, max_bjnz+1, sizeof (int64_t)) ;
        GB_FREE_MEMORY (Coarse_initial, ntasks_initial+1, sizeof (int64_t)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // entire computation in a single coarse task
        //----------------------------------------------------------------------

        TaskList [0].start  = 0 ;
        TaskList [0].end    = bnvec - 1 ;
        TaskList [0].vector = -1 ;
        TaskList [0].hsize  = GB_hash_table_size (flmax, cnrows) ;
        TaskList [0].flops  = total_flops ;
    }

    // free workspace
    GB_FREE_MEMORY (W, ntasks+1, sizeof (int64_t)) ;

    #if GB_TIMING
    double t2 = omp_get_wtime ( ) - tic ; ;
    #endif

#if 0
    // dump the task descriptors
    printf ("\n================== final tasks: ncoarse %d nfine %d ntasks %d\n",
        ncoarse, nfine, ntasks) ;

    for (int fid = 0 ; fid < nfine ; fid++)
    {
        int64_t j  = TaskList [fid].vector ;
        int64_t pB_start = Bp [j] ;
        int64_t p1 = TaskList [fid].start - pB_start ;
        int64_t p2 = TaskList [fid].end   - pB_start ;
        int64_t hsize = TaskList [fid].hsize   ;
        int master = TaskList [fid].master ;
        double work = TaskList [fid].flops ;
        printf ("Fine %3d: ["GBd"] ("GBd" : "GBd") hsize/n %g master %d ",
            fid, j, p1, p2, ((double) hsize) / ((double) cnrows),
            master) ;
        printf (" work %g work/n %g\n", work, work/cnrows) ;
        // if (p1 > p2) printf (":::::::::::::::::: empty task\n") ;
        if (j < 0 || j > cnvec) printf ("j bad\n") ;
    }

    for (int cid = nfine ; cid < ntasks ; cid++)
    {
        int64_t j1 = TaskList [cid].start ;
        int64_t j2 = TaskList [cid].end ;
        int64_t hsize = TaskList [cid].hsize ;
        double work = TaskList [cid].flops ;
        printf ("Coarse %3d: ["GBd" : "GBd"] work/tot: %g hsize/n %g ",
            cid, j1, j2, work / total_flops,
            ((double) hsize) / ((double) cnrows)) ;
        if (cid != TaskList [cid].master) printf ("hey master!\n") ;
        printf (" work %g work/n %g\n", work, work/cnrows) ;
    }

#endif

    #if GB_TIMING
    int nfine_hash = 0 ;
    int nfine_gus = 0 ;
    int ncoarse_hash = 0 ;
    int ncoarse_1hash = 0 ;
    int ncoarse_gus = 0 ;
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t hash_size = TaskList [taskid].hsize ;
        int64_t j = TaskList [taskid].vector ;
        bool is_fine = (j >= 0) ;
        bool use_Gustavson = (hash_size == cnrows) ;
        if (is_fine)
        {
            if (use_Gustavson)
            {
                nfine_gus++ ;
            }
            else
            {
                nfine_hash++ ;
            }
        }
        else
        {
            // coarse task
            int64_t j1 = TaskList [taskid].start ;
            int64_t j2 = TaskList [taskid].end ;
            int64_t nj = j2 - j1 + 1 ;
            if (use_Gustavson)
            {
                ncoarse_gus++ ;
            }
            else if (nj == 1)
            {
                ncoarse_1hash++ ;
            }
            else
            {
                ncoarse_hash++ ;
            }
        }
    }
    #endif

    // Bflops is no longer needed as an alias for Cp
    Bflops = NULL ;

    #if GB_TIMING
    tic = omp_get_wtime ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // allocate the hash tables
    //--------------------------------------------------------------------------

    // If Gustavson's method is used (coarse tasks):
    //
    //      hash_size is cnrows.
    //      Hi is not allocated.
    //      Hf and Hx are both of size hash_size.
    //
    //      (Hf [i] == mark) is true if i is in the hash table.
    //      Hx [i] is the value of C(i,j) during the numeric phase.
    //
    //      Gustavson's method is used if the hash_size for the Hash method
    //      is greater than or equal to cnrows/4.
    //
    // If the Hash method is used (coarse tasks):
    //
    //      hash_size is 2 times the smallest power of 2 that is larger than
    //      the # of flops required for any column C(:,j) being computed.  This
    //      ensures that all entries have space in the hash table, and that the
    //      hash occupancy will never be more than 50%.  It is always smaller
    //      than cnrows/4 (otherwise, Gustavson's method is used).
    //
    //      A hash function is used for the ith entry:
    //          hash = (i * GB_HASH_FACTOR) % hash_size
    //      If a collision occurs, linear probing is used.
    //
    //      (Hf [hash] == mark) is true if the position is occupied.
    //      i = Hi [hash] gives the row index i that occupies that position.
    //      Hx [hash] is the value of C(i,j) during the numeric phase.
    //
    // For both coarse methods:
    //
    //      Hf starts out all zero (via calloc), and mark starts out as 1.  To
    //      clear all of Hf, mark is incremented, so that all entries in Hf are
    //      not equal to mark.

    // add some padding to the end of each hash table, to avoid false
    // sharing of cache lines between the hash tables.
    // TODO: is padding needed?
    #define GB_HASH_PAD (64 / (sizeof (double)))

    int64_t Hi_size_total = 1 ;
    int64_t Hf_size_total = 1 ;
    int64_t Hx_size_total = 1 ;

    // determine the total size of all hash tables
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        if (taskid != TaskList [taskid].master)
        {
            // allocate a single hash table for all fine
            // tasks that compute a single C(:,j)
            continue ;
        }

        int64_t hash_size = TaskList [taskid].hsize ;
        int64_t j = TaskList [taskid].vector ;
        bool is_fine = (j >= 0) ;
        bool use_Gustavson = (hash_size == cnrows) ;
        int64_t j1 = TaskList [taskid].start ;
        int64_t j2 = TaskList [taskid].end ;
        int64_t nj = j2 - j1 + 1 ;

        if (is_fine && use_Gustavson)
        {
            // Hf is uint8_t for the fine Gustavson method, but round up
            // to the nearest number of int64_t values.
            Hf_size_total +=
                GB_CEIL ((hash_size + GB_HASH_PAD), sizeof (int64_t)) ;
        }
        else
        {
            Hf_size_total += (hash_size + GB_HASH_PAD) ;
        }
        if (!is_fine && !use_Gustavson && nj > 1)
        {
            // only large coarse hash tasks need Hi
            Hi_size_total += (hash_size + GB_HASH_PAD) ;
        }
        Hx_size_total += (hash_size + GB_HASH_PAD) ;
    }

    #if 1
    printf ("Hi_size_total %g\n", (double) Hi_size_total) ;
    printf ("Hf_size_total %g\n", (double) Hf_size_total) ;
    printf ("Hx_size_total %g\n", (double) Hx_size_total) ;
    #endif

    // allocate space for all hash tables
    GB_MALLOC_MEMORY (Hi_all, Hi_size_total, sizeof (int64_t)) ;
    GB_CALLOC_MEMORY (Hf_all, Hf_size_total, sizeof (int64_t)) ;
    GB_MALLOC_MEMORY (Hx_all, Hx_size_total, sizeof (double)) ;

    if (0)
    {
        // out of memory
    }

    // split the space into separate hash tables
    int64_t *GB_RESTRICT Hi_split = Hi_all ;
    int64_t *GB_RESTRICT Hf_split = Hf_all ;
    double  *GB_RESTRICT Hx_split = Hx_all ;

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        if (taskid != TaskList [taskid].master)
        {
            // allocate a single hash table for all fine
            // tasks that compute a single C(:,j)
            continue ;
        }

        TaskList [taskid].Hi = Hi_split ;
        TaskList [taskid].Hf = Hf_split ;
        TaskList [taskid].Hx = Hx_split ;

        int64_t hash_size = TaskList [taskid].hsize ;
        int64_t j = TaskList [taskid].vector ;
        bool is_fine = (j >= 0) ;
        bool use_Gustavson = (hash_size == cnrows) ;
        int64_t j1 = TaskList [taskid].start ;
        int64_t j2 = TaskList [taskid].end ;
        int64_t nj = j2 - j1 + 1 ;

        if (is_fine && use_Gustavson)
        {
            // Hf is uint8_t for the fine Gustavson method
            Hf_split += GB_CEIL ((hash_size + GB_HASH_PAD), sizeof (int64_t)) ;
        }
        else
        {
            // Hf is int64_t for all other methods
            Hf_split += (hash_size + GB_HASH_PAD) ;
        }
        if (!is_fine && !use_Gustavson && nj > 1)
        {
            // only coarse hash tasks need Hi
            Hi_split += (hash_size + GB_HASH_PAD) ;
        }
        Hx_split += (hash_size + GB_HASH_PAD) ;
    }

    // assign hash tables to fine task teams
    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int master = TaskList [taskid].master ;
        if (taskid != master)
        {
            TaskList [taskid].Hf = TaskList [master].Hf ;
            TaskList [taskid].Hx = TaskList [master].Hx ;
        }
    }

    #if GB_TIMING
    double t3 = omp_get_wtime ( ) - tic ; ;
    tic = omp_get_wtime ( ) ;
    #endif

    //==========================================================================
    // phase1: count nnz(C(:,j)) for all tasks; do numeric work for fine tasks
    //==========================================================================

    // Coarse tasks: compute nnz (C(:,j1:j2)).
    // Fine tasks: compute nnz (C(:,j)) and values in Hx, via atomics.

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t j = TaskList [taskid].vector ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cnrows) ;
        bool is_fine = (j >= 0) ;

        //----------------------------------------------------------------------
        // do the symbolic task
        //----------------------------------------------------------------------

        if (is_fine)
        {

            //------------------------------------------------------------------
            // fine task: compute nnz (C(:,j)) and values in Hx
            //------------------------------------------------------------------

            int64_t pB_start = TaskList [taskid].start ;
            int64_t pB_end   = TaskList [taskid].end ;
            int64_t my_cjnz = 0 ;
            double *GB_RESTRICT Hx = TaskList [taskid].Hx ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // symbolic+numeric: Gustavson's method for fine task
                //--------------------------------------------------------------

                uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;

                // All of Hf [...] is initially zero.

                // Hf [i] == 0: unlocked, i has not been seen in C(:,j).
                //              Hx [i] is not initialized.
                // Hf [i] == 1: unlocked, i has been seen in C(:,j).
                //              Hx [i] is initialized.
                // Hf [i] == 2: locked.  Hx [i] in an unknown state.

                // TODO: for min, max, and user-defined monoids, the
                // "if (f==1)" test and the if-part must be disabled.

                for (int64_t pB = pB_start ; pB <= pB_end ; pB++)
                {
                    // get B(k,j)
                    int64_t k = Bi [pB] ;
                    double bkj = Bx [pB] ;
                    // scan A(:,k)
                    for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                    {
                        // get A(i,k)
                        int64_t i = Ai [pA] ;
                        double aik = Ax [pA] ;
                        double t = aik * bkj ;          // MULTIPLY
                        int64_t f ;
                        // grab the entry from the hash table
                        #pragma omp atomic read
                        f = Hf [i] ;
                        if (f == 1)
                        {
                            // C(i,j) is already initialized; update it
                            #pragma omp atomic update
                            Hx [i] += t ;           // MONOID
                        }
                        else
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
                                #pragma omp atomic write
                                Hx [i] = t ;
                            }
                            else // f == 1
                            {
                                // C(i,j) already appears in C(:,j)
                                #pragma omp atomic update
                                Hx [i] += t ;           // MONOID
                            }
                            // unlock the entry
                            #pragma omp atomic write
                            Hf [i] = 1 ;
                            if (f == 0)
                            {
                                // a new entry i has been found in C(:,j)
                                my_cjnz++ ;
                            }
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // symbolic+numeric: hash method for fine task
                //--------------------------------------------------------------

                // Each hash entry Hf [hash] splits into two parts, (h,f).  f
                // is the least significant bit.  h is 63 bits, and is the
                // 1-based index i of the C(i,j) entry stored at that location
                // in the hash table.

                // All of Hf [...] is initially zero.

                // Given Hf [hash] split into (h,f):
                // h == 0, f == 0: unlocked, hash entry is unoccupied.
                //                  Hx [hash] is not initialized.
                // h == i+1, f == 0: unlocked, hash entry contains C(i,j).
                //                  Hx [hash] is initialized.
                // h == 0, f == 1: locked, hash entry is unoccupied.
                //                  Hx [hash] is not initialized.
                // h == i+1, f == 1: locked, hash entry contains C(i,j).
                //                  Hx [hash] is initialized.

                int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;

                int64_t hash_bits = (hash_size-1) ;
                for (int64_t pB = pB_start ; pB <= pB_end ; pB++)
                {
                    // get B(k,j)
                    int64_t k = Bi [pB] ;
                    double bkj = Bx [pB] ;
                    // scan A(:,k)
                    for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                    {
                        // get A(i,k)
                        int64_t i = Ai [pA] ;
                        double aik = Ax [pA] ;
                        double t = aik * bkj ;          // MULTIPLY
                        int64_t i1 = i + 1 ;
                        int64_t hf_i_unlocked = (i1 << 1) ;
                        int64_t hf, h, f ;
                        // C(i,j) += A(i,k)*B(k,j)
                        int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                        while (1)
                        {
                            // grab the entry from the hash table
                            #pragma omp atomic read
                            hf = Hf [hash] ;
#if 1
                            // [ only if += atomic update is supported:
                            if (hf == hf_i_unlocked)
                            {
                                // C(i,j) is already initialized; update it.
                                #pragma omp atomic update
                                Hx [hash] += t ;        // MONOID
                                break ;
                            }
                            // ]
#endif
                            h = (hf >> 1) ;
                            if (h == 0 || h == i1)
                            {
                                // Hf [hash] not yet occupied, being
                                // modified by another task, or already
                                // occupied with entry i.  lock the entry.
                                do
                                {
                                    #pragma omp atomic capture
                                    {
                                        hf = Hf [hash] ; Hf [hash] |= 1 ;
                                    }
                                } while (hf & 1) ;
                                if (hf == 0)
                                {
                                    // hash table unoccupied.  claim it.
                                    // C(i,j) has been seen for the first time.
// #pragma omp flush
#pragma omp atomic write
                                    Hx [hash] = t ;
// #pragma omp flush
                                    // unlock the entry
                                    #pragma omp atomic write
                                    Hf [hash] = hf_i_unlocked ;
                                    my_cjnz++ ;
                                    break ;
                                }
                                else if (hf == hf_i_unlocked)
                                {
                                    // entry i is already in this hash entry.
                                    // C(i,j) is already initialized; update it.
// #pragma omp flush
#pragma omp atomic update
                                    Hx [hash] += t ;        // MONOID
// #pragma omp flush
                                    // unlock the entry
                                    #pragma omp atomic write
                                    Hf [hash] = hf_i_unlocked ;
                                    break ;
                                }
                                else
                                {
                                    // Hf [hash] already occupied by different
                                    // entry ; unlock with prior value.
                                    #pragma omp atomic write
                                    Hf [hash] = hf ;
                                }
                            }
                            // linear probing for next entry.
                            hash = (hash + 1) & (hash_bits) ;
                        }
                    }
                }
            }

            TaskList [taskid].my_cjnz = my_cjnz ;

        }
        else
        {

            //------------------------------------------------------------------
            // coarse task: compute nnz in each vector of A*B(:,j1:j2)
            //------------------------------------------------------------------

            int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            int64_t j1 = TaskList [taskid].start ;
            int64_t j2 = TaskList [taskid].end ;
            int64_t mark = 0 ;
            int64_t nj = j2 - j1 + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // symbolic: Gustavson's method for coarse task
                //--------------------------------------------------------------

                for (int64_t j = j1 ; j <= j2 ; j++)
                {
                    // count the entries in C(:,j)
                    int64_t cjnz = 0 ;
                    int64_t bjnz = Bp [j+1] - Bp [j] ;
                    if (bjnz == 1)
                    {
                        // get B(k,j)
                        int64_t k = Bi [Bp [j]] ;
                        // C(:,j) = A(:,k)*B(k,j) for a single entry B(k,j)
                        cjnz = Ap [k+1] - Ap [k] ;
                    }
                    else if (bjnz > 1)
                    {
                        mark++ ;
                        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                        {
                            // get B(k,j)
                            int64_t k = Bi [pB] ;
                            // scan A(:,k)
                            for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                            {
                                // get A(i,k)
                                int64_t i = Ai [pA] ;
                                // add i to the gather/scatter workspace
                                if (Hf [i] != mark)
                                {
                                    // C(i,j) is a new entry
                                    Hf [i] = mark ;
                                    cjnz++ ;
                                }
                            }
                        }
                    }
                    Cp [j] = cjnz ;
                }

            }
            else if (nj == 1)
            {

                //--------------------------------------------------------------
                // symbolic: 1-column coarse hash task
                //--------------------------------------------------------------

                // Hi is not used.  Hf [hash] is zero if the hash entry is
                // empty, or ((i+1) << 1) if it contains entry i.

                int64_t hash_bits = (hash_size-1) ;
                int64_t j = j1 ;

                // count the entries in C(:,j)
                int64_t cjnz = 0 ;
                int64_t bjnz = Bp [j+1] - Bp [j] ;
                if (bjnz == 1)
                {
                    // get B(k,j)
                    int64_t k = Bi [Bp [j]] ;
                    // C(:,j) = A(:,k)*B(k,j) for a single entry B(k,j)
                    cjnz = Ap [k+1] - Ap [k] ;
                }
                else if (bjnz > 1)
                {
                    for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                    {
                        // get B(k,j)
                        int64_t k = Bi [pB] ;
                        // scan A(:,k)
                        int64_t pA_end = Ap [k+1] ;
                        for (int64_t pA = Ap [k] ; pA < pA_end ; pA++)
                        {
                            // get A(i,k)
                            int64_t i = Ai [pA] ;
                            int64_t i1 = (i+1) << 1 ;
                            // find i in the hash table
                            int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                            while (1)
                            {
                                int64_t h = Hf [hash] ;
                                if (h == i1)
                                {
                                    // already in the hash table
                                    break ;
                                }
                                else if (h == 0)
                                {
                                    // C(i,j) is a new entry.
                                    // hash entry is not occupied;
                                    // add i to the hash table
                                    Hf [hash] = i1 ;
                                    cjnz++ ;
                                    break ;
                                }
                                else
                                {
                                    // linear probing
                                    hash = (hash + 1) & (hash_bits) ;
                                }
                            }
                        }
                    }
                }
                Cp [j] = cjnz ;

            }
            else
            {

                //--------------------------------------------------------------
                // symbolic: large coarse hash task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;

                int64_t hash_bits = (hash_size-1) ;
                for (int64_t j = j1 ; j <= j2 ; j++)
                {
                    // count the entries in C(:,j)
                    int64_t cjnz = 0 ;
                    int64_t bjnz = Bp [j+1] - Bp [j] ;
                    if (bjnz == 1)
                    {
                        // get B(k,j)
                        int64_t k = Bi [Bp [j]] ;
                        // C(:,j) = A(:,k)*B(k,j) for a single entry B(k,j)
                        cjnz = Ap [k+1] - Ap [k] ;
                    }
                    else if (bjnz > 1)
                    {
                        mark++ ;
                        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                        {
                            // get B(k,j)
                            int64_t k = Bi [pB] ;
                            // scan A(:,k)
                            int64_t pA_end = Ap [k+1] ;
                            for (int64_t pA = Ap [k] ; pA < pA_end ; pA++)
                            {
                                // get A(i,k)
                                int64_t i = Ai [pA] ;
                                // find i in the hash table
                                int64_t hash = (i * GB_HASH_FACTOR)
                                               & (hash_bits) ;
                                while (1)
                                {
                                    if (Hf [hash] == mark)
                                    {
                                        // hash entry is occupied
                                        if (Hi [hash] == i)
                                        {
                                            // i already in the hash table
                                            break ;
                                        }
                                        else
                                        {
                                            // linear probing
                                            hash = (hash + 1) & (hash_bits) ;
                                        }
                                    }
                                    else
                                    {
                                        // C(i,j) is a new entry.
                                        // hash entry is not occupied;
                                        // add i to the hash table
                                        Hf [hash] = mark ;
                                        Hi [hash] = i ;
                                        cjnz++ ;
                                        break ;
                                    }
                                }
                            }
                        }
                    }
                    Cp [j] = cjnz ;
                }
            }
        }
    }

    #if GB_TIMING
    double t4 = omp_get_wtime ( ) - tic ; ;
    tic = omp_get_wtime ( ) ;
    #endif

    //==========================================================================
    // compute Cp with cumulative sum
    //==========================================================================

    // TaskList [taskid].my_cjnz is the # of unique entries found in C(:,j) by
    // that task.  Sum these terms to compute total # of entries in C(:,j).

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int64_t j = TaskList [taskid].vector ;
        Cp [j] = 0 ;
    }

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int64_t j = TaskList [taskid].vector ;
        int64_t my_cjnz = TaskList [taskid].my_cjnz ;
        Cp [j] += my_cjnz ;
        ASSERT (my_cjnz <= cnrows) ;
    }

    // Cp [j] is now nnz (C (:,j)), for all vectors j, whether computed by fine
    // tasks or coarse tasks.

    GB_cumsum (Cp, cncols, nonempty_result, nthreads) ;
    int64_t cnz = Cp [cncols] ;

    //==========================================================================
    // allocate Ci and Cx
    //==========================================================================

    GB_MALLOC_MEMORY (Ci, cnz, sizeof (int64_t)) ;
    GB_MALLOC_MEMORY (Cx, cnz, sizeof (double)) ;

    if (0)
    {
        // out of memory
    }

    #if GB_TIMING
    double t5 = omp_get_wtime ( ) - tic ;
    tic = omp_get_wtime ( ) ;
    #endif

    //==========================================================================
    // phase2: numeric phase for coarse tasks, prep for gather for fine tasks
    //==========================================================================

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        double  *GB_RESTRICT Hx = TaskList [taskid].Hx ;
        int64_t j = TaskList [taskid].vector ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cnrows) ;
        bool is_fine = (j >= 0) ;

        //----------------------------------------------------------------------
        // do the numeric task
        //----------------------------------------------------------------------

        if (is_fine)
        {

            //------------------------------------------------------------------
            // count nnz (C(:,j) for the final gather for this fine task
            //------------------------------------------------------------------

            int nfine_team_size = TaskList [taskid].nfine_team_size ;
            int master     = TaskList [taskid].master ;
            int my_teamid  = taskid - master ;
            int64_t my_cjnz = 0 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // final numeric: Gustavson's method for fine task
                //--------------------------------------------------------------

                int64_t pC = Cp [j] ;
                int64_t cjnz = Cp [j+1] - pC ;
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cnrows, my_teamid, nfine_team_size);

                if (cjnz == cnrows)
                {

                    //----------------------------------------------------------
                    // C(:,j) is entirely dense: finish the work now
                    //----------------------------------------------------------

                    for (int64_t i = istart ; i < iend ; i++)
                    {
                        Ci [pC + i] = i ;
                    }
                    memcpy (Cx + pC + istart, Hx + istart,
                        (iend-istart) * sizeof (double)) ;

                }
                else
                {

                    //----------------------------------------------------------
                    // C(:,j) is sparse: count the work for this fine task
                    //----------------------------------------------------------

                    // TODO this is slow if cjnz << cnrows

                    uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;

                    // O(cnrows) linear scan of Hf to create the pattern of
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
                // final numeric: hash method for fine task
                //--------------------------------------------------------------

                // TODO this is slow if cjnz << hash_size

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
            // numeric coarse task: compute C(:,j1:j2)
            //------------------------------------------------------------------

            int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
            int64_t j1 = TaskList [taskid].start ;
            int64_t j2 = TaskList [taskid].end ;
            int64_t nj = j2 - j1 + 1 ;
            int64_t mark = nj + 1 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // numeric: Gustavson's method for coarse task
                //--------------------------------------------------------------

                for (int64_t j = j1 ; j <= j2 ; j++)
                {

                    //----------------------------------------------------------
                    // compute the pattern and values of C(:,j)
                    //----------------------------------------------------------

                    int64_t bjnz = Bp [j+1] - Bp [j] ;
                    if (bjnz == 0)
                    {
                        // nothing to do
                        continue ;
                    }

                    int64_t pC = Cp [j] ;
                    int64_t cjnz = Cp [j+1] - pC ;

                    if (bjnz == 1)
                    {

                        //------------------------------------------------------
                        // C(:,j) = A(:,k)*B(k,j) for a single entry B(k,j)
                        //------------------------------------------------------

                        int64_t pB = Bp [j] ;
                        // get B(k,j)
                        int64_t k  = Bi [pB] ;
                        double bkj = Bx [pB] ;
                        // scan A(:,k)
                        for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                        {
                            // get A(i,k)
                            int64_t i  = Ai [pA] ;
                            double aik = Ax [pA] ;
                            double t = aik * bkj ;      // MULTIPLY
                            // update C(i,j) in gather/scatter work
                            // C(i,j) = A(i,k) * B(k,j)
                            Cx [pC] = t ;
                            // log the row index in C(:,j)
                            Ci [pC] = i ;
                            pC++ ;
                        }

                    }
                    else if (cjnz == cnrows)
                    {

                        //------------------------------------------------------
                        // C(:,j) is dense; compute directly in Ci and Cx
                        //------------------------------------------------------

                        for (int64_t i = 0 ; i < cnrows ; i++)
                        {
                            Ci [pC + i] = i ;
                            Cx [pC + i] = 0 ;
                        }

                        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                        {
                            // get B(k,j)
                            int64_t k  = Bi [pB] ;
                            double bkj = Bx [pB] ;
                            // scan A(:,k)
                            for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                            {
                                // get A(i,k)
                                int64_t i  = Ai [pA] ;
                                double aik = Ax [pA] ;
                                // C(i,j) += A(i,k) * B(k,j)
                                Cx [pC + i] += aik * bkj ;  // MULTIPLY, MONOID
                            }
                        }

                    }
                    else if (cjnz > cnrows / 16)
                    {

                        //------------------------------------------------------
                        // C(:,j) is not very sparse
                        //------------------------------------------------------

                        // compute C(:,j) in gather/scatter workspace
                        mark++ ;
                        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                        {
                            // get B(k,j)
                            int64_t k  = Bi [pB] ;
                            double bkj = Bx [pB] ;
                            // scan A(:,k)
                            for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                            {
                                // get A(i,k)
                                int64_t i  = Ai [pA] ;
                                double aik = Ax [pA] ;
                                double t = aik * bkj ;      // MULTIPLY
                                // update C(i,j) in gather/scatter workspace
                                if (Hf [i] != mark)
                                {
                                    // C(i,j) = A(i,k) * B(k,j)
                                    Hf [i] = mark ;
                                    Hx [i] = t ;
                                }
                                else
                                {
                                    // C(i,j) += A(i,k) * B(k,j)
                                    Hx [i] += t ;           // MONOID
                                }
                            }
                        }

                        // gather the pattern and values into C(:,j)
                        for (int64_t i = 0 ; i < cnrows ; i++)
                        {
                            if (Hf [i] == mark)
                            {
                                Ci [pC] = i ;
                                Cx [pC] = Hx [i] ;
                                pC++ ;
                            }
                        }

                    }
                    else
                    {

                        //------------------------------------------------------
                        // C(:,j) is very sparse
                        //------------------------------------------------------

                        // compute C(:,j) in gather/scatter workspace
                        mark++ ;
                        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                        {
                            // get B(k,j)
                            int64_t k  = Bi [pB] ;
                            double bkj = Bx [pB] ;
                            // scan A(:,k)
                            for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                            {
                                // get A(i,k)
                                int64_t i  = Ai [pA] ;
                                double aik = Ax [pA] ;
                                double t = aik * bkj ;      // MULTIPLY
                                // update C(i,j) in gather/scatter work
                                if (Hf [i] != mark)
                                {
                                    // C(i,j) = A(i,k) * B(k,j)
                                    Hf [i] = mark ;
                                    Hx [i] = t ;
                                    // log the row index in C(:,j)
                                    Ci [pC++] = i ;
                                }
                                else
                                {
                                    // C(i,j) = A(i,k) * B(k,j)
                                    Hx [i] += t ;           // MONOID
                                }
                            }
                        }

                        // sort the pattern of C(:,j)
                        GB_qsort_1a (Ci + Cp [j], cjnz) ;   // coarse Gustavson

                        // gather the values into C(:,j)
                        for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
                        {
                            int64_t i = Ci [p] ;
                            Cx [p] = Hx [i] ;
                        }
                    }
                }

            }
            else if (nj == 1)
            {

                //--------------------------------------------------------------
                // 1-column coarse hash task
                //--------------------------------------------------------------

                int64_t hash_bits = (hash_size-1) ;
                int64_t j = j1 ;

                int64_t bjnz = Bp [j+1] - Bp [j] ;
                if (bjnz == 0)
                {
                    // nothing to do
                    continue ;
                }

                int64_t pC = Cp [j] ;
                int64_t cjnz = Cp [j+1] - pC ;

                if (bjnz == 1)
                {
                    int64_t pB = Bp [j] ;
                    // get B(k,j)
                    int64_t k  = Bi [pB] ;
                    double bkj = Bx [pB] ;
                    // scan A(:,k)
                    for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                    {
                        // get A(i,k)
                        int64_t i  = Ai [pA] ;
                        double aik = Ax [pA] ;
                        double t = aik * bkj ;      // MULTIPLY
                        // update C(i,j) in gather/scatter work
                        // C(i,j) = A(i,k) * B(k,j)
                        Cx [pC] = t ;
                        // log the row index in C(:,j)
                        Ci [pC] = i ;
                        pC++ ;
                    }

                }
                else
                {

                    // Hf [hash] has been set to (i+1)<<1 in the symbolic
                    // phase, for all entries i in the pattern of C(:,j).
                    // The first time Hf [hash] is seen, it is incremented
                    // to ((i+1)<<1)+1 to denote it has been seen.

                    for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                    {
                        // get B(k,j)
                        int64_t k  = Bi [pB] ;
                        double bkj = Bx [pB] ;
                        // scan A(:,k)
                        int64_t pA_end = Ap [k+1] ;
                        for (int64_t pA = Ap [k] ; pA < pA_end ; pA++)
                        {
                            // get A(i,k)
                            int64_t i  = Ai [pA] ;
                            int64_t i1 = (i+1) << 1 ;
                            int64_t i2 = i1 + 1 ;
                            double aik = Ax [pA] ;
                            double t = aik * bkj ;      // MULTIPLY
                            // find i in the hash table
                            int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                            while (1)
                            {
                                int64_t h = Hf [hash] ;
                                if (h == i2)
                                {
                                    // C(i,j) has been seen before; update it.
                                    // C(i,j) += A(i,k) * B(k,j)
                                    Hx [hash] += t ; // MONOID
                                    break ;
                                }
                                else if (h == i1)
                                {
                                    // first time C(i,j) seen in numeric phase
                                    // C(i,j) = A(i,k) * B(k,j)
                                    Hf [hash] = i2 ;
                                    Hx [hash] = t ;
                                    Ci [pC++] = i ;
                                    break ;
                                }
                                else
                                {
                                    // linear probing
                                    hash = (hash + 1) & (hash_bits) ;
                                }
                            }
                        }
                    }

                    // sort the pattern of C(:,j)
                    GB_qsort_1a (Ci + Cp [j], cjnz) ;   // coarse hash

                    // gather the values of C(:,j)
                    for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
                    {
                        int64_t i = Ci [p] ;
                        // find C(i,j) in the hash table
                        int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                        int64_t i2 = ((i+1) << 1) + 1 ;
                        while (1)
                        {
                            if (Hf [hash] == i2)
                            {
                                // i found in the hash table
                                Cx [p] = Hx [hash] ;
                                break ;
                            }
                            else
                            {
                                // linear probing
                                hash = (hash + 1) & (hash_bits) ;
                            }
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // hash method for large coarse task
                //--------------------------------------------------------------

                int64_t *GB_RESTRICT Hi = TaskList [taskid].Hi ;

                int64_t hash_bits = (hash_size-1) ;

                for (int64_t j = j1 ; j <= j2 ; j++)
                {

                    //----------------------------------------------------------
                    // compute the pattern and values of C(:,j)
                    //----------------------------------------------------------

                    int64_t bjnz = Bp [j+1] - Bp [j] ;
                    if (bjnz == 0)
                    {
                        // nothing to do
                        continue ;
                    }

                    int64_t pC = Cp [j] ;
                    int64_t cjnz = Cp [j+1] - pC ;

                    if (bjnz == 1)
                    {

                        //------------------------------------------------------
                        // C(:,j) = A(:,k)*B(k,j) for a single entry B(k,j)
                        //------------------------------------------------------

                        int64_t pB = Bp [j] ;
                        // get B(k,j)
                        int64_t k  = Bi [pB] ;
                        double bkj = Bx [pB] ;
                        // scan A(:,k)
                        for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                        {
                            // get A(i,k)
                            int64_t i  = Ai [pA] ;
                            double aik = Ax [pA] ;
                            double t = aik * bkj ;      // MULTIPLY
                            // update C(i,j) in gather/scatter work
                            // C(i,j) = A(i,k) * B(k,j)
                            Cx [pC] = t ;
                            // log the row index in C(:,j)
                            Ci [pC] = i ;
                            pC++ ;
                        }

                    }
                    else
                    {

                        //------------------------------------------------------
                        // compute C(:,j) using the hash method
                        //------------------------------------------------------

                        mark++ ;
                        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
                        {
                            // get B(k,j)
                            int64_t k  = Bi [pB] ;
                            double bkj = Bx [pB] ;
                            // scan A(:,k)
                            int64_t pA_end = Ap [k+1] ;
                            for (int64_t pA = Ap [k] ; pA < pA_end ; pA++)
                            {
                                // get A(i,k)
                                int64_t i  = Ai [pA] ;
                                double aik = Ax [pA] ;
                                double t = aik * bkj ;      // MULTIPLY
                                // find i in the hash table
                                int64_t hash = (i * GB_HASH_FACTOR)
                                               & (hash_bits) ;
                                while (1)
                                {
                                    if (Hf [hash] == mark)
                                    {
                                        // hash entry is occupied
                                        if (Hi [hash] == i)
                                        {
                                            // i already in the hash table,
                                            // at Hi [hash]
                                            // C(i,j) += A(i,k) * B(k,j)
                                            Hx [hash] += t ; // MONOID
                                            break ;
                                        }
                                        else
                                        {
                                            // linear probing
                                            hash = (hash + 1) & (hash_bits) ;
                                        }
                                    }
                                    else
                                    {
                                        // hash entry is not occupied;
                                        // add i to the hash table
                                        // C(i,j) = A(i,k) * B(k,j)
                                        Hf [hash] = mark ;
                                        Hi [hash] = i ;
                                        Hx [hash] = t ;
                                        Ci [pC++] = i ;
                                        break ;
                                    }
                                }
                            }
                        }

                        // sort the pattern of C(:,j)
                        GB_qsort_1a (Ci + Cp [j], cjnz) ;   // coarse hash

                        // gather the values of C(:,j)
                        for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
                        {
                            int64_t i = Ci [p] ;
                            // find C(i,j) in the hash table
                            int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                            while (1)
                            {
                                if (Hi [hash] == i)
                                {
                                    // i found in the hash table
                                    Cx [p] = Hx [hash] ;
                                    break ;
                                }
                                else
                                {
                                    // linear probing
                                    hash = (hash + 1) & (hash_bits) ;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #if GB_TIMING
    double t6 = omp_get_wtime ( ) - tic ; ;

    #if 0
    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int64_t j = TaskList [taskid].vector ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cnrows) ;
        bool is_fine = (j >= 0) ;
        if (is_fine)
        {
            int64_t my_cjnz = TaskList [taskid].my_cjnz ;
            double fl       = TaskList [taskid].flops ;
            printf ("fine count %d: my_cjnz "GBd"\n", taskid, my_cjnz) ;
        }
    }
    #endif

    tic = omp_get_wtime ( ) ;
    #endif

    // free workspace
    GB_FREE_MEMORY (Hi_all, Hi_size_total, sizeof (int64_t)) ;

    //==========================================================================
    // final numeric gather phase for fine tasks
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
            bool use_Gustavson = (hash_size == cnrows) ;
            if (!use_Gustavson)
            {
                int64_t j = TaskList [taskid].vector ;
                int64_t cjnz = Cp [j+1] - Cp [j] ;
                cjnz_max = GB_IMAX (cjnz_max, cjnz) ;
            }
        }
        int64_t my_cjnz = TaskList [taskid].my_cjnz ;
        TaskList [taskid].my_cjnz = cjnz_sum ;
        cjnz_sum += my_cjnz ;
    }

    #if GB_TIMING
    printf ("cjnz_max "GBd"\n", cjnz_max) ;
    #endif

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t j = TaskList [taskid].vector ;
        double  *GB_RESTRICT Hx = TaskList [taskid].Hx ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cnrows) ;

        int64_t pC = Cp [j] ;
        int64_t cjnz = Cp [j+1] - pC ;
        pC += TaskList [taskid].my_cjnz ;

        int nfine_team_size = TaskList [taskid].nfine_team_size ;
        int master     = TaskList [taskid].master ;
        int my_teamid  = taskid - master ;

        //----------------------------------------------------------------------
        // gather the values into C(:,j)
        //----------------------------------------------------------------------

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // final numeric gather: Gustavson's method for fine task
            //------------------------------------------------------------------

            // TODO this is slow if cjnz << cnrows

            uint8_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;

            if (cjnz < cnrows)
            {
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, cnrows, my_teamid, nfine_team_size);

                for (int64_t i = istart ; i < iend ; i++)
                {
                    if (Hf [i])
                    {
                        Ci [pC] = i ;
                        Cx [pC] = Hx [i] ;
                        pC++ ;
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // final numeric gather: hash method for fine task
            //------------------------------------------------------------------

            // TODO this is slow if cjnz << hash_size

            int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;

            int64_t hash_start, hash_end ;
            GB_PARTITION (hash_start, hash_end, hash_size,
                my_teamid, nfine_team_size) ;

            for (int64_t hash = hash_start ; hash < hash_end ; hash++)
            {
                int64_t hf = Hf [hash] ;
                if (hf != 0)
                {
                    int64_t i = (hf >> 1) - 1 ;
                    Ci [pC++] = i ;
                }
            }
        }
    }

    #if GB_TIMING
    double t7 = omp_get_wtime ( ) - tic ; ;
    tic = omp_get_wtime ( ) ;
    #endif

    //==========================================================================
    // final numeric gather phase for fine tasks (hash method)
    //==========================================================================

    if (cjnz_max > 0)
    {
        bool parallel_sort = (cjnz_max > GB_BASECASE && nthreads_max > 1) ;

        // allocate workspace
        if (parallel_sort)
        {
            GB_MALLOC_MEMORY (W, cjnz_max, sizeof (int64_t)) ;
            if (0)
            {
                // out of memory
            }
        }

        for (int taskid = 0 ; taskid < nfine ; taskid++)
        {
            int64_t hash_size  = TaskList [taskid].hsize ;
            bool use_Gustavson = (hash_size == cnrows) ;

            if (!use_Gustavson && taskid == TaskList [taskid].master)
            {
                int64_t j = TaskList [taskid].vector ;
                int64_t hash_bits = (hash_size-1) ;
                int64_t *GB_RESTRICT Hf = TaskList [taskid].Hf ;
                double  *GB_RESTRICT Hx = TaskList [taskid].Hx ;
                int64_t cjnz = Cp [j+1] - Cp [j] ;

                //--------------------------------------------------------------
                // sort the pattern of C(:,j)
                //--------------------------------------------------------------

                int nth = GB_nthreads (cjnz, chunk, nthreads_max) ;

                #if GB_TIMING
                double t9 = omp_get_wtime ( ) ;
                #endif

                if (parallel_sort && nth > 1)
                {
                    // parallel mergesort
                    GB_msort_1 (Ci + Cp [j], W, cjnz, nth) ;   // fine hash
                }
                else
                {
                    // sequential quicksort
                    GB_qsort_1a (Ci + Cp [j], cjnz) ;   // fine hash
                }

                #if GB_TIMING
                t9 = omp_get_wtime ( ) - t9 ;
                printf ("sort %d: j "GBd" cjnz "GBd" time: %g threads: %d\n",
                    taskid, j, cjnz, t9, nth) ;
                #endif

                //--------------------------------------------------------------
                // gather the values of C(:,j)
                //--------------------------------------------------------------

                #pragma omp parallel for num_threads(nth) schedule(static)
                for (int64_t pC = Cp [j] ; pC < Cp [j+1] ; pC++)
                {
                    // get C(i,j)
                    int64_t i = Ci [pC] ;
                    int64_t i1 = i + 1 ;
                    // find i in the hash table
                    int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                    while (1)
                    {
                        // Hf is not modified, so no atomic read is needed.
                        int64_t hf ;
                        {
                            hf = Hf [hash] ;
                        }
                        int64_t h = (hf >> 1) ;
                        if (h == i1)
                        {
                            // i already in the hash table, at Hf [hash]
                            Cx [pC] = Hx [hash] ;
                            break ;
                        }
                        else
                        {
                            // linear probing
                            hash = (hash + 1) & (hash_bits) ;
                        }
                    }
                }
            }
        }

        // free workspace
        GB_FREE_MEMORY (W, cjnz_max, sizeof (int64_t)) ;
    }

    //==========================================================================
    // free workspace and return result
    //==========================================================================

    #if GB_TIMING
    double t8 = omp_get_wtime ( ) - tic ; ;
    double tot = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 ;

    printf ("ntasks %d ncoarse %d (gus: %d onehash: %d hash: %d)"
        " nfine %d (gus: %d hash: %d)\n", ntasks,
        ncoarse, ncoarse_gus, ncoarse_1hash, ncoarse_hash,
        nfine, nfine_gus, nfine_hash) ;

    printf ("t1: fl time        %16.6f (%10.2f)\n", t1, 100*t1/tot) ;
    printf ("t2: task time      %16.6f (%10.2f)\n", t2, 100*t2/tot) ;
    printf ("t3: alloc H time   %16.6f (%10.2f)\n", t3, 100*t3/tot) ;
    printf ("t4: phase1         %16.6f (%10.2f)\n", t4, 100*t4/tot) ;
    printf ("t5: cumsum time    %16.6f (%10.2f)\n", t5, 100*t5/tot) ;
    printf ("t6: phase2         %16.6f (%10.2f)\n", t6, 100*t6/tot) ;
    printf ("t7: fine gather1   %16.6f (%10.2f)\n", t7, 100*t7/tot) ;
    printf ("t8: fine gather2   %16.6f (%10.2f)\n", t8, 100*t8/tot) ;

    printf ("   total time                                 [[[ %g ]]]\n", tot) ;
    printf ("   total flops %g\n", (double) total_flops) ;
    #endif

    (*Cp_handle) = Cp ;
    (*Ci_handle) = Ci ;
    (*Cx_handle) = Cx ;

    GB_FREE_MEMORY (TaskList, ntasks, sizeof (GB_hashtask_struct)) ;
    GB_FREE_MEMORY (Hf_all, Hf_size_total, sizeof (int64_t)) ;
    GB_FREE_MEMORY (Hx_all, Hx_size_total, sizeof (double)) ;

    return (0) ;
}

















    //--------------------------------------------------------------------------
    // C<M> = A'*B, via masked dot product method and built-in semiring
    //--------------------------------------------------------------------------

    bool done = false ;

#ifndef GBCOMPACT

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    #define GB_Adot3B(add,mult,xyname) GB_Adot3B_ ## add ## mult ## xyname

    #define GB_AxB_WORKER(add,mult,xyname)                              \
    {                                                                   \
        info = GB_Adot3B (add,mult,xyname) (C, M,                       \
            A, A_is_pattern, B, B_is_pattern,                           \
            TaskList, ntasks, nthreads) ;                               \
        done = (info != GrB_NO_VALUE) ;                                 \
    }                                                                   \
    break ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    GB_Opcode mult_opcode, add_opcode ;
    GB_Type_code xycode, zcode ;

    if (GB_AxB_semiring_builtin (A, A_is_pattern, B, B_is_pattern, semiring,
        flipxy, &mult_opcode, &add_opcode, &xycode, &zcode))
    { 
        #include "GB_AxB_factory.c"
    }

#endif

    //--------------------------------------------------------------------------
    // user semirings created at compile time
    //--------------------------------------------------------------------------

    if (semiring->object_kind == GB_USER_COMPILED)
    { 
        // determine the required type of A and B for the user semiring
        GrB_Type atype_required, btype_required ;

        if (flipxy)
        { 
            // A is passed as y, and B as x, in z = mult(x,y)
            atype_required = mult->ytype ;
            btype_required = mult->xtype ;
        }
        else
        { 
            // A is passed as x, and B as y, in z = mult(x,y)
            atype_required = mult->xtype ;
            btype_required = mult->ytype ;
        }

        if (A->type == atype_required && B->type == btype_required)
        {
            info = GB_AxB_user (GxB_AxB_DOT, semiring, Chandle, M, A, B,
                flipxy,
                /* heap: */ NULL, NULL, NULL, 0,
                /* Gustavson: */ NULL,
                /* dot2: */ NULL, NULL, nthreads, 0, 0, NULL,
                /* dot3: */ TaskList, ntasks) ;
            done = true ;
        }
    }

    //--------------------------------------------------------------------------
    // C = A*B, via saxpy3 method and typecasting
    //--------------------------------------------------------------------------

    if (!done)
    {

        //----------------------------------------------------------------------
        // get operators, functions, workspace, contents of A, B, C, and M
        //----------------------------------------------------------------------

        GxB_binary_function fmult = mult->function ;
        GxB_binary_function fadd  = add->op->function ;

        size_t csize = C->type->size ;
        size_t asize = A_is_pattern ? 0 : A->type->size ;
        size_t bsize = B_is_pattern ? 0 : B->type->size ;

        size_t xsize = mult->xtype->size ;
        size_t ysize = mult->ytype->size ;

        // scalar workspace: because of typecasting, the x/y types need not
        // be the same as the size of the A and B types.
        // flipxy false: aki = (xtype) A(k,i) and bkj = (ytype) B(k,j)
        // flipxy true:  aki = (ytype) A(k,i) and bkj = (xtype) B(k,j)
        size_t aki_size = flipxy ? ysize : xsize ;
        size_t bkj_size = flipxy ? xsize : ysize ;

        GB_void *GB_RESTRICT terminal = add->terminal ;

        GB_cast_function cast_A, cast_B ;
        if (flipxy)
        { 
            // A is typecasted to y, and B is typecasted to x
            cast_A = A_is_pattern ? NULL : 
                     GB_cast_factory (mult->ytype->code, A->type->code) ;
            cast_B = B_is_pattern ? NULL : 
                     GB_cast_factory (mult->xtype->code, B->type->code) ;
        }
        else
        { 
            // A is typecasted to x, and B is typecasted to y
            cast_A = A_is_pattern ? NULL :
                     GB_cast_factory (mult->xtype->code, A->type->code) ;
            cast_B = B_is_pattern ? NULL :
                     GB_cast_factory (mult->ytype->code, B->type->code) ;
        }

        //----------------------------------------------------------------------
        // C<M> = A'*B via dot products, function pointers, and typecasting
        //----------------------------------------------------------------------

        // aki = A(k,i), located in Ax [pA]
        #define GB_GETA(aki,Ax,pA)                                          \
            GB_void aki [GB_VLA(aki_size)] ;                                \
            if (!A_is_pattern) cast_A (aki, Ax +((pA)*asize), asize) ;

        // bkj = B(k,j), located in Bx [pB]
        #define GB_GETB(bkj,Bx,pB)                                          \
            GB_void bkj [GB_VLA(bkj_size)] ;                                \
            if (!B_is_pattern) cast_B (bkj, Bx +((pB)*bsize), bsize) ;

        // break if cij reaches the terminal value
        #define GB_DOT_TERMINAL(cij)                                        \
            if (terminal != NULL && memcmp (cij, terminal, csize) == 0)     \
            {                                                               \
                break ;                                                     \
            }

        // C(i,j) = A(i,k) * B(k,j)
        #define GB_MULT(cij, aki, bkj)                                      \
            GB_MULTIPLY (cij, aki, bkj) ;                                   \

        // C(i,j) += A(i,k) * B(k,j)
        #define GB_MULTADD(cij, aki, bkj)                                   \
            GB_void zwork [GB_VLA(csize)] ;                                 \
            GB_MULTIPLY (zwork, aki, bkj) ;                                 \
            fadd (cij, cij, zwork) ;

        // define cij for each task
        #define GB_CIJ_DECLARE(cij)                                         \
            GB_void cij [GB_VLA(csize)] ;

        // address of Cx [p]
        #define GB_CX(p) Cx +((p)*csize)

        // save the value of C(i,j)
        #define GB_CIJ_SAVE(cij,p)                                          \
            memcpy (GB_CX (p), cij, csize) ;

        #define GB_ATYPE GB_void
        #define GB_BTYPE GB_void
        #define GB_CTYPE GB_void

        // loops with function pointers cannot be vectorized
        #define GB_DOT_SIMD ;

        if (flipxy)
        { 
            #define GB_MULTIPLY(z,x,y) fmult (z,y,x)
            #include "GB_AxB_dot3_template.c"
            #undef GB_MULTIPLY
        }
        else
        { 
            #define GB_MULTIPLY(z,x,y) fmult (z,x,y)
            #include "GB_AxB_dot3_template.c"
            #undef GB_MULTIPLY
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    ASSERT_MATRIX_OK (C, "saxpy3: C = A*B output", GB0) ;
    ASSERT (*Chandle == C) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    return (GrB_SUCCESS) ;
}

