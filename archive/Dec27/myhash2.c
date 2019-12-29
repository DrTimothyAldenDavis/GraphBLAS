//------------------------------------------------------------------------------
// GB_AxB_hash:  C = A*B using a mix of Gustavson's and Hash methods
//------------------------------------------------------------------------------

#define GB_DEBUG 1

#include "myhash.h"
#include "GB_sort.h"

#define GB_NTASKS_PER_THREAD 2
#define GB_HASH_FACTOR 107
#define GB_TIMING 1
#define GB_COSTLY 2

//------------------------------------------------------------------------------
// hash table entries
//------------------------------------------------------------------------------

// For fine tasks, the hash table entry Hf [hash] can be marked in three
// different ways.  The first 62 bits are used to encode an integer for the
// 3rd phase, which is the matrix index place in that hash entry.

#define GB_FINE_MARK_1ST (0x01)
#define GB_FINE_MARK_2ND (0x02)
#define GB_FINE_MARK_BITS (0x03)

//------------------------------------------------------------------------------
// GB_hashtask_struct: task descriptor for GB_AxB_hash
//------------------------------------------------------------------------------

// A coarse task computes C(:,j1:j2) = A*B(:,j1:j2), for a contiguous set of
// vectors j1:j2.  A coarse taskid is denoted byTaskList [taskid].vector == -1,
// j1 = TaskList [taskid].start, and j2 = TaskList [taskid].end.  No summation
// is needed for the final result of each coarse task.

// A fine taskid computes A*B(k1:k2,j) for a single vector C(:,j), for a
// contiguous range k1:k2, where j = Tasklist[taskid].vector (which is >= 0),
// k1 = Bi [TaskList [taskid].start], k2 = Bi [TaskList [taskid].end].  It sums
// its computations in a hash table shared by all fine tasks that compute
// C(:,j).

// Both tasks use a hash table allocated uniquely for the task, in Hi, Hf, and
// Hx.  The size of the hash table is determined by the maximum # of flops
// needed to compute any vector in C(:,j1:j2) for a coarse task, or the
// entire computation of the single vector in a fine task.  The table has
// a size that is twice the smallest a power of 2 larger than the flop count.
// If this size is >= cnrows/4, then the Gustavson method is not used, and
// the hash size is set to cnrows.

typedef struct
{
    int64_t start ;     // starting vector for coarse task, p for fine task
    int64_t end ;       // ending vector for coarse task, p for fine task
    int64_t vector ;    // -1 for coarse task, vector j for fine task
    int64_t hsize ;     // size of hash table
    int64_t *Hi ;       // Hi array for hash table (NULL for Gustavson's method)
    int64_t *Hf ;       // Hf array for hash table
    double  *Hx ;       // Hx array for hash table
    int64_t cp ;        // pointer to Cp [j] for fine task
    int master ;        // master task of a fine task
}
GB_hashtask_struct ;

//------------------------------------------------------------------------------
// GB_hash_table_size
//------------------------------------------------------------------------------

// flmax is the max flop count for computing A*B(:,j), for any column j that
// this task computes.  GB_hash_table_size determines the hash table size for
// this task, which is twice the smallest power of 2 larger than flmax.  If
// flmax is large enough, the hash_size is returned as cnrows, so that
// Gustavson's method will be used instead of the Hash method.

static inline int64_t GB_hash_table_size
(
    int64_t flmax,
    int64_t cnrows
)
{
    // hash_size = 2 * (smallest power of 2 that is >= to flmax)
    double hlog = log2 ((double) flmax) ;
    int64_t hash_size = ((int64_t) 2) << ((int64_t) floor (hlog) + 1) ;
    // use Gustavson's method if hash_size is too big
    bool use_Gustavson = (hash_size >= cnrows/16) ;
    if (use_Gustavson)
    {
        hash_size = cnrows ;
    }
    return (hash_size) ;
}

//------------------------------------------------------------------------------
// GB_create_coarse_task: create a single coarse task
//------------------------------------------------------------------------------

// Compute the max flop count for any vector in a coarse task, determine the
// hash table size, and construct the coarse task.

static inline void GB_create_coarse_task
(
    int64_t j1,         // coarse task consists of vectors j1:j2
    int64_t j2,
    GB_hashtask_struct *TaskList,
    int taskid,         // taskid for this coarse task
    int64_t *Bflops,    // size bnvec; cum sum of flop counts for vectors of B
    int64_t cnrows,     // # of rows of B and C
    double chunk,
    int nthreads_max
)
{
    // find the max # of flops for any vector in this task
    int64_t flmax = 1 ;
    int nth = GB_nthreads (j2-j1+1, chunk, nthreads_max) ;
    #pragma omp parallel for num_threads(nth) schedule(static) \
        reduction(max:flmax)
    for (int64_t j = j1 ; j <= j2 ; j++)
    {
        int64_t fl = Bflops [j+1] - Bflops [j] ;
        flmax = GB_IMAX (flmax, fl) ;
    }
    // define the coarse task
    TaskList [taskid].start  = j1 ;
    TaskList [taskid].end    = j2 ;
    TaskList [taskid].vector = -1 ;
    TaskList [taskid].master = taskid ;
    TaskList [taskid].hsize  =  GB_hash_table_size (flmax, cnrows) ;
}

//------------------------------------------------------------------------------

int64_t myhash2
(

    int64_t **Cp_handle,
    int64_t **Ci_handle,
    double  **Cx_handle,

    int64_t *restrict Ap,
    int64_t *restrict Ai,
    double  *restrict Ax,
    int64_t anrows,
    int64_t ancols,

    int64_t *restrict Bp,
    int64_t *restrict Bi,
    double  *restrict Bx,
    int64_t bnrows,
    int64_t bncols,

    int64_t *nonempty_result
)
{

    printf ("\n---------- parallel version 2\n") ;

    int64_t *GB_RESTRICT Hi_all = NULL ;
    int64_t *GB_RESTRICT Hf_all = NULL ;
    double  *GB_RESTRICT Hx_all = NULL ;
    int64_t *GB_RESTRICT Coarse_initial = NULL ;    // initial coarse tasks
    GB_hashtask_struct *GB_RESTRICT TaskList = NULL ;
    int64_t *GB_RESTRICT Cp = NULL ;
    int64_t *GB_RESTRICT Ci = NULL ;
    double  *GB_RESTRICT Cx = NULL ;

    int64_t *GB_RESTRICT W = NULL ;
    int64_t *GB_RESTRICT Bflops2    = NULL ;

    //--------------------------------------------------------------------------
    // determine # of threads available
    //--------------------------------------------------------------------------

    int nthreads ;
    GxB_get (GxB_NTHREADS, &nthreads) ;

    // GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads_max = nthreads ;
    double chunk = 4096 ;

    #if GB_TIMING
    double tic = omp_get_wtime ( ) ;
    // double tsort = 0 ;
    int nquick = 0 ;
    #endif

//    printf ("nthreads %d GB_COSTLY %d\n", nthreads, GB_COSTLY) ;

    //--------------------------------------------------------------------------
    // get problem size and allocate Cp
    //--------------------------------------------------------------------------

    int64_t cnrows = anrows ;
    int64_t cncols = bncols ;
    int64_t bnvec  = bncols ;
    int64_t cnvec  = bncols ;
    GB_MALLOC_MEMORY (Cp, cnvec+1, sizeof (int64_t)) ;

    if (0)
    {
        // out of memory
    }

    //==========================================================================
    // compute flop counts for each vector of B and C
    //==========================================================================

    int64_t *restrict Bflops = Cp ;     // Cp is used as workspace for Bflops 
    int64_t flmax = 1 ;
    int64_t total_flops = 0 ;

    int nth = GB_nthreads (bnvec, chunk, nthreads_max) ;

    // TODO use GB_AxB_flopcount
    #pragma omp parallel for num_threads(nth) schedule(guided) \
        reduction(max:flmax) reduction(+:total_flops)
    for (int64_t j = 0 ; j < bnvec ; j++)
    {
        // fl = flop count to compute C(:,j) = A*B(:j)
        int64_t fl = 0 ;
        for (int64_t pB = Bp [j] ; pB < Bp [j+1] ; pB++)
        {
            // get B(k,j)
            int64_t k = Bi [pB] ;
            // flop count for A*B(k,j)
            fl += (Ap [k+1] - Ap [k]) ;
        }
        if (nthreads > 1)
        {
            // keep the flop count if creating parallel tasks
            Bflops [j] = fl ;
        }
        flmax = GB_IMAX (flmax, fl) ;
        total_flops += fl ;
    }

    #if GB_TIMING
    double t1 = omp_get_wtime ( ) - tic ; ;
    printf ("t1: fl time %g\n", t1) ;
    tic = omp_get_wtime ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // determine # of threads and # of initial coarse tasks
    //--------------------------------------------------------------------------

    nthreads = GB_nthreads ((double) total_flops, chunk, nthreads) ;
    int ntasks_initial = (nthreads == 1) ?
        1 : (GB_NTASKS_PER_THREAD * nthreads) ;
    double target_task_size = ((double) total_flops) / ntasks_initial ;

    //==========================================================================
    // determine # of parallel tasks
    //==========================================================================

    int nfine = 0 ;         // # of fine tasks
    int ncoarse = 0 ;       // # of coarse tasks
    int64_t max_bjnz = 0 ;  // max (nnz (B (:,j))) of fine tasks

    if (ntasks_initial > 1)
    {

        //----------------------------------------------------------------------
        // cumulative sum of flop counts for each vector of B and C
        //----------------------------------------------------------------------

        // FUTURE: possible int64_t overflow
        GB_cumsum (Bflops, bnvec, NULL, nth) ;

        //----------------------------------------------------------------------
        // construct initial coarse tasks
        //----------------------------------------------------------------------

        if (!GB_pslice (&Coarse_initial, Bflops, bnvec, ntasks_initial))
        {
            // out of memory
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
                        int nfine_this = ceil (jflops / target_task_size) ;
                        nfine += nfine_this ;
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

    printf ("ntasks %d ncoarse %d nfine %d\n", ntasks, ncoarse, nfine) ;

    //==========================================================================
    // allocate workspace to construct fine tasks
    //==========================================================================

    GB_CALLOC_MEMORY (TaskList, ntasks, sizeof (GB_hashtask_struct)) ;
    GB_MALLOC_MEMORY (W, ntasks+1, sizeof (int64_t)) ;
    if (nfine > 0)
    {
        GB_MALLOC_MEMORY (Bflops2, max_bjnz+1, sizeof (int64_t)) ;
    }

    if (0)
    {
        // out of memory
    }

    //==========================================================================
    // create the tasks
    //==========================================================================

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
                        int nfine_this = ceil (jflops / target_task_size) ;
                        GB_pslice (&W, Bflops2, bjnz, nfine_this) ;

                        // hash table for all fine takes for C(:,j)
                        int64_t hsize = GB_hash_table_size (jflops, cnrows) ;

                        // construct the fine tasks for B(:,j)
                        int master = nf ;
                        for (int fid = 0 ; fid < nfine_this ; fid++)
                        {
                            int64_t pstart = W [fid] ;
                            int64_t pend   = W [fid+1] ;
                            int64_t fl = Bflops2 [pend] - Bflops2 [pstart] ;
                            TaskList [nf].start  = pB_start + pstart ;
                            TaskList [nf].end    = pB_start + pend - 1 ;
                            TaskList [nf].vector = j ;
                            TaskList [nf].hsize  = hsize ;
                            TaskList [nf].master = master ;
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
    }

    #if GB_TIMING
    double t2 = omp_get_wtime ( ) - tic ; ;
    printf ("t2: task time %g\n", t2) ;
    #endif

#if 1
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
        printf ("Fine %3d: ["GBd"] ("GBd" : "GBd") hsize/n %g master %d\n",
            fid, j, p1, p2, ((double) hsize) / ((double) cnrows), 
            master) ;
        // if (p1 > p2) printf (":::::::::::::::::: empty task\n") ;
        if (j < 0 || j > cnvec) printf ("j bad\n") ;
    }

    for (int cid = nfine ; cid < ntasks ; cid++)
    {
        int64_t j1 = TaskList [cid].start ;
        int64_t j2 = TaskList [cid].end ;
        double work = (nthreads == 1) ? total_flops :
            (Bflops [j2+1] - Bflops [j1]) ;
        int64_t hsize = TaskList [cid].hsize ;
        printf ("Coarse %3d: ["GBd" : "GBd"] work/tot: %g hsize/n %g\n",
            cid, j1, j2, work / total_flops,
            ((double) hsize) / ((double) cnrows)) ;
        if (cid != TaskList [cid].master) printf ("hey master!\n") ;
    }
#endif

    // Bflops is no longer needed as an alias for Cp
    Bflops = NULL ;

    #if GB_TIMING
    tic = omp_get_wtime ( ) ;
    #endif

    //==========================================================================
    // allocate the hash tables
    //==========================================================================

    // If Gustavson's method is used:
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
    // If the Hash method is used:
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
    // For both methods:
    //
    //      Hf starts out all zero (via calloc), and mark starts out as 1.  To
    //      clear all of Hf, mark is incremented, so that all entries in Hf are
    //      not equal to mark.

    // add some padding to the end of each hash table, to avoid false
    // sharing of cache lines between the hash tables.
    // TODO: is padding needed?
    #define GB_HASH_PAD (64 / (sizeof (double)))

    int64_t Hi_size_total = 1 ;
    int64_t Hx_size_total = 1 ;

    // determine the total size of all hash tables
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        if (taskid != TaskList [taskid].master)
        {
            // allocate a single hash table for all fine
            // tasks that compute a single C(:,j)
            // printf ("task %d is fine slave\n", taskid) ;
            continue ;
        }

        int64_t hsize = TaskList [taskid].hsize ;
        int64_t j = TaskList [taskid].vector ;
        bool is_fine = (j >= 0) ;
        if (hsize < cnrows && !is_fine)
        {
            // only coarse hash tasks need Hi
            Hi_size_total += (hsize + GB_HASH_PAD) ;
        }
        Hx_size_total += (hsize + GB_HASH_PAD) ;
    }

    // allocate space for all hash tables
    GB_MALLOC_MEMORY (Hi_all, Hi_size_total, sizeof (int64_t)) ;
    GB_CALLOC_MEMORY (Hf_all, Hx_size_total, sizeof (int64_t)) ;
    GB_MALLOC_MEMORY (Hx_all, Hx_size_total, sizeof (double)) ;

    if (0)
    {
        // out of memory
    }

    // split the space into separate hash tables
    int64_t *restrict Hi = Hi_all ;
    int64_t *restrict Hf = Hf_all ;
    double  *restrict Hx = Hx_all ;

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        if (taskid != TaskList [taskid].master)
        {
            // allocate a single hash table for all fine
            // tasks that compute a single C(:,j)
            // printf ("task %d is fine slave\n", taskid) ;
            continue ;
        }

        TaskList [taskid].Hi = Hi ;
        TaskList [taskid].Hf = Hf ;
        TaskList [taskid].Hx = Hx ;

        int64_t hsize = TaskList [taskid].hsize ;
        int64_t j = TaskList [taskid].vector ;
        bool is_fine = (j >= 0) ;
        if (hsize < cnrows && !is_fine)
        {
            // only coarse hash tasks need Hi
            Hi += (hsize + GB_HASH_PAD) ;
        }
        Hf += (hsize + GB_HASH_PAD) ;
        Hx += (hsize + GB_HASH_PAD) ;
    }

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int master = TaskList [taskid].master ;
        if (taskid != master)
        {
            // printf ("get hash for %d fine slave\n", taskid) ;
            TaskList [taskid].Hi = TaskList [master].Hi ;
            TaskList [taskid].Hf = TaskList [master].Hf ;
            TaskList [taskid].Hx = TaskList [master].Hx ;
        }
    }

    //==========================================================================
    // symbolic phase: count # of entries in each vector of C
    //==========================================================================

    // Coarse tasks: compute nnz (C(:,j1:j2))
    // Fine tasks: compute nnz (C(:,j)) via atomics

    #if GB_TIMING
    int nfine_hash = 0 ;
    int ncoarse_hash = 0 ;
    #endif

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t *restrict Hf = TaskList [taskid].Hf ;
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
            // fine task: compute nnz (C(:,j)) for this task
            //------------------------------------------------------------------

            int64_t pB_start = TaskList [taskid].start ;
            int64_t pB_end   = TaskList [taskid].end ;
            int64_t my_cjnz = 0 ;
            double *restrict Hx = TaskList [taskid].Hx ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // symbolic: Gustavson's method for fine task
                //--------------------------------------------------------------

                // uint8_t *restrict Hff = Hf ;

                for (int64_t pB = pB_start ; pB <= pB_end ; pB++)
                {
                    // get B(k,j)
                    int64_t k = Bi [pB] ;
                    // scan A(:,k)
                    for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                    {
                        // get A(i,k)
                        int64_t i = Ai [pA] ;
                        // parallel: atomic swap
                        // int64_t v ;
                        int64_t v ;
                        #pragma omp atomic read
                        v = Hf [i] ;

                        if (v != GB_FINE_MARK_1ST)
                        {
                            #pragma omp atomic capture
                            {
                                v = Hf [i] ; Hf [i] = GB_FINE_MARK_1ST ;
                                // v = Hff [i] ; Hff [i] = GB_FINE_MARK_1ST ;
                            }
                            if (v != GB_FINE_MARK_1ST)
                            {
                                // increment # unique entries from this task
                                my_cjnz++ ;
                                #pragma omp atomic write
                                Hx [i] = 0 ;
                            }
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // symbolic: hash method for fine task
                //--------------------------------------------------------------

                #if GB_TIMING
                nfine_hash++ ;
                #endif

                int64_t hash_bits = (hash_size-1) ;
                for (int64_t pB = pB_start ; pB <= pB_end ; pB++)
                {
                    // get B(k,j)
                    int64_t k = Bi [pB] ;
                    // scan A(:,k)
                    for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                    {
                        // get A(i,k)
                        int64_t i = Ai [pA] ;
                        // find i in the hash table
                        int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                        while (1)
                        {
                            // parallel: atomic read and modify
                            int64_t v ;
                            #pragma omp atomic capture
                            {
                                v = Hf [hash] ; Hf [hash] |= GB_FINE_MARK_1ST ;
                            }
                            if ((v & GB_FINE_MARK_BITS) == GB_FINE_MARK_1ST)
                            {
                                // hash entry is already occuppied.  It might
                                // be in the process of being modified by the
                                // task that owns the entry.  Spin-wait until
                                // the other tasks writes its value x, in the
                                // atomic write below.
                                while (v == GB_FINE_MARK_1ST)
                                {
                                    // parallel: atomic read
                                    #pragma omp atomic read
                                    v = Hf [hash] ;
                                }
                                int64_t h = (v >> 2) - 1 ;
                                if (h == i)
                                {
                                    // i already in the hash table, at Hf [hash]
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
                                // hash entry is not occuppied;
                                // add i to the hash table at this location
                                int64_t x = ((i+1) << 2) | GB_FINE_MARK_1ST ;
                                // parallel: atomic write
                                {
                                    #pragma omp atomic write
                                    Hf [hash] = x ;
                                }
                                // increment # unique entries from this task
                                my_cjnz++ ;
                                break ;
                            }
                        }
                    }
                }
            }

//          printf ("task %d j "GBd " my_cjnz "GBd"\n", taskid, j, my_cjnz) ;
            TaskList [taskid].cp = my_cjnz ;

        }
        else
        {

            //------------------------------------------------------------------
            // coarse task: compute nnz in each vector of A*B(:,j1:j2)
            //------------------------------------------------------------------

            int64_t j1 = TaskList [taskid].start ;
            int64_t j2 = TaskList [taskid].end ;
            int64_t mark = 0 ;

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
                        #if GB_TIMING
                        nquick++ ;
                        #endif
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
                                    Hf [i] = mark ;
                                    cjnz++ ;
                                }
                            }
                        }
                    }
                    Cp [j] = cjnz ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // symbolic: hash method for coarse task
                //--------------------------------------------------------------

                int64_t *restrict Hi = TaskList [taskid].Hi ;

                #if GB_TIMING
                ncoarse_hash++ ;
                #endif

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
                        #if GB_TIMING
                        nquick++ ;
                        #endif
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
                                // find i in the hash table
                                int64_t hash = (i * GB_HASH_FACTOR)
                                               & (hash_bits) ;
                                while (1)
                                {
                                    if (Hf [hash] == mark)
                                    {
                                        // hash entry is occuppied
                                        int64_t h = Hi [hash] ;
                                        if (h == i)
                                        {
                                            // i already in the hash table,
                                            // at Hi [hash]
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
                                        // hash entry is not occuppied;
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
    double t3 = omp_get_wtime ( ) - tic ; ;
    printf ("t3: sym time %g   (%d %d):%d\n", t3, ncoarse_hash, nfine_hash,
        nquick) ;
//  for (int taskid = 0 ; taskid < nfine ; taskid++)
//  {
//      int64_t j = TaskList [taskid].vector ;
//      int64_t my_cjnz = TaskList [taskid].cp ;
//      printf ("task %d j "GBd" my_cjnz "GBd"\n", taskid, j, my_cjnz) ;
//  }
    tic = omp_get_wtime ( ) ;
    nquick = 0 ;
    #endif

    // Now all hash table entries used in any computation contain
    // GB_FINE_MARK_1ST, in their 2 least significant bits.  This is used in
    // the next phase, to implement a spin-wait lock for the atomic monoid.

    //==========================================================================
    // compute Cp with cumulative sum
    //==========================================================================

    // TaskList [taskid].cp is the # of unique entries found in C(:,j) by that
    // task.  Sum these terms to compute total # of entries in C(:,j).

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int64_t j = TaskList [taskid].vector ;
        Cp [j] = 0 ;
    }

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int64_t j = TaskList [taskid].vector ;
        int64_t my_cjnz = TaskList [taskid].cp ;
        Cp [j] += my_cjnz ;
        ASSERT (my_cjnz <= cnrows) ;
    }

    // Cp [j] is now nnz (C (:,j)), for all vectors j, whether computed by fine
    // tasks or coarse tasks.

    GB_cumsum (Cp, cncols, nonempty_result, nthreads) ;
    int64_t cnz = Cp [cncols] ;

    // Cp [j] is now the vector pointers for C.  Copy the C(:,j) pointer back
    // into each fine task.

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        if (taskid == TaskList [taskid].master)
        {
            // this is the master fine task for C(:,j)
            int64_t j = TaskList [taskid].vector ;
            TaskList [taskid].cp = Cp [j] ;
        }
    }

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
    double t4 = omp_get_wtime ( ) - tic ;
    printf ("t4: cumsum time %g\n", t4) ;
    tic = omp_get_wtime ( ) ;
    #endif

    //==========================================================================
    // numerical phase
    //==========================================================================

    #if GB_TIMING
    nfine_hash = 0 ;
    ncoarse_hash = 0 ;
    #endif

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t *restrict Hf = TaskList [taskid].Hf ;
        double  *restrict Hx = TaskList [taskid].Hx ;
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
            // numeric fine task: compute C(:,j), leave numeric result in Hx
            //------------------------------------------------------------------

            int64_t pB_start = TaskList [taskid].start ;
            int64_t pB_end   = TaskList [taskid].end ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // numeric: Gustavson's method for fine task
                //--------------------------------------------------------------

                // uint8_t *restrict Hff = Hf ;

                for (int64_t pB = pB_start ; pB <= pB_end ; pB++)
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
                        double t = aik * bkj ;          // MULTIPLY

                        #if 0
                        // lock the critical section for ith entry
                        uint8_t v ;
                        do
                        {
                            #pragma omp atomic capture
                            {
                                v = Hf [i] ; Hf [i] = 0 ;
                                // v = Hff [i] ; Hff [i] = 0 ;
                            }
                            // if v is != 0 then this task has acquired the
                            // lock, and has set the lock to 0.  If v is zero,
                            // then another task is in the critical section,
                            // and the lock is still 0.
                        } while (v == 0) ;
                        if (v == GB_FINE_MARK_1ST)
                        {
                            // C(i,j) is a new entry in C(:,j)
                            Hx [i] = t ;
                        }
                        else
                        {
                            // C(i,j) already appears in C(:,j)
                            Hx [i] += t ;           // MONOID
                        }
                        // unlock the critical section for ith entry
                        #pragma omp atomic write
                        Hf [i] = GB_FINE_MARK_2ND ;
                        // Hff [i] = GB_FINE_MARK_2ND ;
                        #else

                            #pragma omp atomic update
                            Hx [i] += t ;           // MONOID

                        #endif
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // numeric: hash method for fine task
                //--------------------------------------------------------------

                // pC is shared with all fine tasks that compute C(:,j)
                int master = TaskList [taskid].master ;
                int64_t *restrict pC = &(TaskList [master].cp) ;

                #if GB_TIMING
                nfine_hash++ ;
                #endif

                int64_t hash_bits = (hash_size-1) ;
                for (int64_t pB = pB_start ; pB <= pB_end ; pB++)
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
                        // C(i,j) += A(i,k)*B(k,j)
                        int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                        while (1)
                        {
                            int64_t v ;
                            {
                                #pragma omp atomic read
                                v = Hf [hash] ;
                            }
                            int64_t h = (v >> 2) - 1 ;
                            if (h == i)
                            {
                                // i in the hash table, at Hf [hash]
                                // parallel: atomic update
                                int64_t x = v & (~GB_FINE_MARK_BITS) ;
                                // lock the critical section for ith entry
                                do
                                {
                                    #pragma omp atomic capture
                                    {
                                        v = Hf [hash] ; Hf [hash] = x ;
                                    }
                                    // if the 2 least significant bits of v are
                                    // nonzero, then this task has acquired the
                                    // lock, and set those 2 bits to 0.
                                    // Otherwise, another task is in the
                                    // critical section.
                                } while (v == x) ;
                                // C(i,j) += t
                                if ((v & GB_FINE_MARK_BITS) == GB_FINE_MARK_1ST)
                                {
                                    // C(i,j) is a new entry in the hash table
                                    Ci [(*pC)++] = i ;
                                    Hx [hash] = t ;
                                }
                                else
                                {
                                    // C(i,j) already appears in the hash table
                                    Hx [hash] += t ;
                                }
                                // unlock the critical section for ith entry
                                x = x | GB_FINE_MARK_2ND ;
                                #pragma omp atomic write
                                Hf [hash] = x ;
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
        else
        {

            //------------------------------------------------------------------
            // numeric coarse task: compute C(:,j1:j2)
            //------------------------------------------------------------------

            int64_t j1 = TaskList [taskid].start ;
            int64_t j2 = TaskList [taskid].end ;
            int64_t mark = j2 - j1 + 2 ;

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

                        #if GB_TIMING
                        nquick++ ;
                        #endif

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
            else
            {

                //--------------------------------------------------------------
                // hash method for coarse task
                //--------------------------------------------------------------

                int64_t *restrict Hi = TaskList [taskid].Hi ;

                #if GB_TIMING
                ncoarse_hash++ ;
                #endif
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

                        #if GB_TIMING
                        nquick++ ;
                        #endif

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
                            for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
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
                                        // hash entry is occuppied
                                        int64_t h = Hi [hash] ;
                                        if (h == i)
                                        {
                                            // i already in the hash table,
                                            // at Hi [hash]
                                            // C(i,j) += A(i,k) * B(k,j)
                                            Hx [hash] += t ;    // MONOID
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
                                        // hash entry is not occuppied;
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
                                int64_t h = Hi [hash] ;
                                if (h == i)
                                {
                                    // i already in the hash table, at Hi [hash]
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
    double t5 = omp_get_wtime ( ) - tic ; ;
    printf ("t5: num time %g   (%d):%d\n", t5, nfine_hash, nquick) ;
    tic = omp_get_wtime ( ) ;
    nquick = 0 ;
    #endif

    //==========================================================================
    // final numeric phase: gather work for fine tasks
    //==========================================================================

    #if GB_TIMING
    nfine_hash = 0 ;
    int nfine_1 = 0 ;
    int nfine_2 = 0 ;
    #endif

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        if (taskid != TaskList [taskid].master)
        {
            // only the master fine task does the gather for C(:,j)
            continue ;
        }

        int64_t j = TaskList [taskid].vector ;
        int64_t *restrict Hf = TaskList [taskid].Hf ;
        double  *restrict Hx = TaskList [taskid].Hx ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cnrows) ;
//      printf ("task %d hash_size "GBd" cnrows "GBd" ratio %g\n",
//          taskid, hash_size, cnrows,
//          (double) hash_size / (double) cnrows) ;

        int64_t pC = Cp [j] ;
        int64_t cjnz = Cp [j+1] - pC ;
        int nth = GB_nthreads (cnrows, chunk, nthreads_max) ;

        //----------------------------------------------------------------------
        // gather the values into C(:,j)
        //----------------------------------------------------------------------

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // final numeric: Gustavson's method for fine task
            //------------------------------------------------------------------

            if (cjnz == cnrows)
            {

                //--------------------------------------------------------------
                // C(:,j) is entirely dense
                //--------------------------------------------------------------

                // TODO prior phase could compute its result in Ci and Cx
                // itself.

                #if GB_TIMING
                nfine_1++ ;
                #endif

                #pragma omp parallel for num_threads(nth) schedule(static)
                for (int64_t i = 0 ; i < cnrows ; i++)
                {
                    Ci [pC + i] = i ;
                }
                GB_memcpy (Cx + pC, Hx, cnrows * sizeof (double), nth) ;

            }
            else
            {

                #if GB_TIMING
                nfine_2++ ;
                #endif

                //--------------------------------------------------------------
                // C(:,j) is sparse
                //--------------------------------------------------------------

                // O(cnrows) linear scan of Hf to create the pattern of C(:,j).
                // No sort is needed.

                // uint8_t *restrict Hff = Hf ;

                if (nth == 1)
                {
                    for (int64_t i = 0 ; i < cnrows ; i++)
                    {
                        // if (Hf [i] == GB_FINE_MARK_2ND)
                        // if (Hf [i] == GB_FINE_MARK_1ST)
                        // if (Hff [i])
                        if (Hf [i])
                        {
                            Ci [pC] = i ;
                            Cx [pC] = Hx [i] ;
                            pC++ ;
                        }
                    }
                }
                else
                {

                    #pragma omp parallel for num_threads(nth) schedule(static)
                    for (int tid = 0 ; tid < nth ; tid++)
                    {
                        int64_t my_cjnz = 0, istart, iend ;
                        GB_PARTITION (istart, iend, cnrows, tid, nth) ;
                        for (int64_t i = istart ; i < iend ; i++)
                        {
                            if (Hf [i])
                            // if (Hf [i] == GB_FINE_MARK_2ND)
                            // if (Hf [i] == GB_FINE_MARK_1ST)
                            // if (Hff [i])
                            {
                                my_cjnz++ ;
                            }
                        }
                        W [tid] = my_cjnz ;
                    }

                    GB_cumsum (W, nth, NULL, 1) ;

                    #pragma omp parallel for num_threads(nth) schedule(static)
                    for (int tid = 0 ; tid < nth ; tid++)
                    {
                        int64_t p = pC + W [tid], istart, iend ;
                        GB_PARTITION (istart, iend, cnrows, tid, nth) ;
                        for (int64_t i = istart ; i < iend ; i++)
                        {
                            if (Hf [i])
                            // if (Hf [i] == GB_FINE_MARK_2ND)
                            // if (Hf [i] == GB_FINE_MARK_1ST)
                            // if (Hff [i])
                            {
                                Ci [p] = i ;
                                Cx [p] = Hx [i] ;
                                p++ ;
                            }
                        }
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // final numeric: hash method for fine task
            //------------------------------------------------------------------

            int64_t hash_bits = (hash_size-1) ;

            // sort the pattern of C(:,j)

            // TODO use a parallel sort
            GB_qsort_1a (Ci + Cp [j], cjnz) ;   // fine hash

            #if GB_TIMING
            nfine_hash++ ;
            #endif

            #pragma omp parallel for num_threads(nth) schedule(static)
            for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
            {
                // get C(i,j)
                int64_t i = Ci [p] ;
                // find i in the hash table
                int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                while (1)
                {
                    // Hf is not modified, so no atomic read is needed.
                    int64_t v ;
                    {
                        v = Hf [hash] ;
                    }
                    int64_t h = (v >> 2) - 1 ;
                    if (h == i)
                    {
                        // i already in the hash table, at Hf [hash]
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

    #if GB_TIMING
    double t6 = omp_get_wtime ( ) - tic ;
    printf ("t6: num2 time %g  (%d %d %d)\n", t6, nfine_1, nfine_2, nfine_hash);
    double tot = t1 + t2 + t3 + t4 + t5 + t6 ;
    printf ("   t1 %10.2f (compute flop counts)\n", 100 * t1/tot) ;
    printf ("   t2 %10.2f (create tasks)\n", 100 * t2/tot) ;
    printf ("   t3 %10.2f (sym)\n", 100 * t3/tot) ;
    printf ("   t4 %10.2f (cumsum)\n", 100 * t4/tot) ;
    printf ("   t5 %10.2f (num)\n", 100 * t5/tot) ;
    printf ("   t6 %10.2f (num2)\n", 100 * t6/tot) ;
    printf ("   total time                                 ::: %g :::\n", tot) ;
    printf ("   total flops %g\n", (double) total_flops) ;
    #endif

    //==========================================================================
    // free workspace and return result
    //==========================================================================

    (*Cp_handle) = Cp ;
    (*Ci_handle) = Ci ;
    (*Cx_handle) = Cx ;

    GB_FREE_MEMORY (TaskList, ntasks, sizeof (GB_hashtask_struct)) ;
    GB_FREE_MEMORY (W, ntasks+1, sizeof (int64_t)) ;
    GB_FREE_MEMORY (Hi_all, Hi_size_total, sizeof (int64_t)) ;
    GB_FREE_MEMORY (Hf_all, Hx_size_total, sizeof (int64_t)) ;
    GB_FREE_MEMORY (Hx_all, Hx_size_total, sizeof (double)) ;

    return (0) ;
}

