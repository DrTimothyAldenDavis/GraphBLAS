//------------------------------------------------------------------------------
// GB_AxB_hash:  C = A*B using a mix of Gustavson's and Hash methods
//------------------------------------------------------------------------------

#include "myhash.h"
#include "GB_sort.h"

#define GB_NTASKS_PER_THREAD 4
#define GB_HASH_FACTOR 107
#define GB_TIMING 0

//------------------------------------------------------------------------------
// hash table entries
//------------------------------------------------------------------------------

// For fine tasks, the hash table entry Hf [hash] can be marked in three
// different ways.  The first 62 bits are used to encode an integer for the
// 3rd phase, which is the matrix index place in that hash entry.

#define GB_FINE_MARK_1ST (0x01)
#define GB_FINE_MARK_2ND (0x02)
#define GB_FINE_MARK_3RD (0x03)

//------------------------------------------------------------------------------
// GB_hashtask_struct: task descriptor for GB_AxB_hash
//------------------------------------------------------------------------------

// A coarse task computes C(:,j1:j2) = A*B(:,j1:j2), for a contiguous set of
// vectors j1:j2.  A coarse taskid is denoted byTaskList [taskid].vector == -1,
// j1 = TaskList [taskid].start, and j2 = TaskList [taskid].end.  No summation
// is needed for the final result of each coarse task.

// A fine taskid computes F{taskid} = A*B(k1:k2,j) for a single vector C(:,j),
// for a contiguous range k1:k2, where j = Tasklist[taskid].vector (which is >=
// 0), k1 = Bi [TaskList [taskid].start], k2 = Bi [TaskList [taskid].end], and
// fjnz = nnz (F{taskid}).  Once all F terms are computed, they are summed
// together from each fine task to compute C(:,j).

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
    int64_t *Fi ;       // pattern of F = A*B(k1:k2,j) for a fine task
    double  *Fx ;       // values of F for a fine task
    int64_t fjnz ;      // nnz for C(:,j), fine task
    int64_t cp ;        // pointer to Cp [j] for fine task
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
    bool use_Gustavson = (hash_size >= cnrows/4) ;
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
    int64_t cnrows      // # of rows of B and C
)
{
    // find the max # of flops for any vector in this task
    int64_t flmax = 1 ;
    // #pragma omp parallel for schedule(static) reduction(max:flmax)
    for (int64_t j = j1 ; j <= j2 ; j++)
    {
        int64_t fl = Bflops [j+1] - Bflops [j] ;
        flmax = GB_IMAX (flmax, fl) ;
    }
    // define the coarse task
    TaskList [taskid].start  = j1 ;
    TaskList [taskid].end    = j2 ;
    TaskList [taskid].vector = -1 ;
    TaskList [taskid].hsize  =  GB_hash_table_size (flmax, cnrows) ;
}

//------------------------------------------------------------------------------

int64_t myhash
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

    //--------------------------------------------------------------------------
    // determine # of threads available
    //--------------------------------------------------------------------------

    int nthreads ;
    GxB_get (GxB_NTHREADS, &nthreads) ;

    #if GB_TIMING
    double tic [2] ;
    simple_tic (tic) ;
    double tsort = 0 ;
    double tic2 [2] ;
    int nquick = 0 ;
    #endif

    //--------------------------------------------------------------------------
    // get problem size and allocate Cp
    //--------------------------------------------------------------------------

    int64_t cnrows = anrows ;
    int64_t cncols = bncols ;
    int64_t bnvec  = bncols ;
    int64_t cnvec  = bncols ;
    int64_t *restrict Cp = mxMalloc ((cnvec + 1) * sizeof (int64_t)) ;

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

    // #pragma omp parallel for schedule(guided) reduction(max:flmax)
    //      reduction(+:total_flops)
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
            // keep the flop count if
            Bflops [j] = fl ;
        }
        flmax = GB_IMAX (flmax, fl) ;
        total_flops += fl ;
    }

    #if GB_TIMING
    double t1 = simple_toc (tic) ;
    printf ("t1: fl time %g\n", t1) ;
    simple_tic (tic) ;
    #endif

    //--------------------------------------------------------------------------
    // determine # of threads and # of initial coarse tasks
    //--------------------------------------------------------------------------

    nthreads = GB_nthreads ((double) total_flops, 4096 /* chunk */, nthreads) ;
    int ntasks_initial = (nthreads == 1) ?
        1 : (GB_NTASKS_PER_THREAD * nthreads) ;
    double target_task_size = ((double) total_flops) / ntasks_initial ;

    //==========================================================================
    // determine # of parallel tasks
    //==========================================================================

    int64_t *Coarse_initial = NULL ;    // initial set of coarse tasks
    GB_hashtask_struct *TaskList = NULL ;
    int nfine = 0 ;         // # of fine tasks
    int ncoarse = 0 ;       // # of coarse tasks
    int64_t max_bjnz = 0 ;  // max (nnz (B (:,j))) of fine tasks

    if (ntasks_initial > 1)
    {

        //----------------------------------------------------------------------
        // cumulative sum of flop counts for each vector of B and C
        //----------------------------------------------------------------------

        // FUTURE: possible int64_t overflow
        GB_cumsum (Bflops, bnvec, NULL, nthreads) ;

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
            else if (task_flops > 4 * target_task_size)
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

                    if (jflops > 2 * target_task_size && bjnz > 1)
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

    int64_t *Fine_slice = NULL ;
    int64_t *Bflops2    = NULL ;
    TaskList = mxCalloc ((ntasks+1), sizeof (GB_hashtask_struct)) ;
    if (nfine > 0)
    {
        Fine_slice = mxMalloc ((nfine+1) * sizeof (int64_t)) ;
        Bflops2    = mxMalloc ((max_bjnz + 1) * sizeof (int64_t)) ;
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
            else if (task_flops > 4 * target_task_size)
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

                    if (jflops > 2 * target_task_size && bjnz > 1)
                    {
                        // A*B(:,j) is costly; split it into 2 or more fine
                        // tasks.  First flush the prior coarse task, if any.
                        if (jcoarse_start < j)
                        {
                            // jcoarse_start:j-1 form a single coarse task
                            GB_create_coarse_task (jcoarse_start, j-1,
                                TaskList, nc++, Bflops, cnrows) ;
                        }

                        // next coarse task (if any) starts at j+1
                        jcoarse_start = j+1 ;

                        // count the work for each B(k,j)
                        int64_t pB_start = Bp [j] ;
                        // #pragma omp parallel for
                        for (int64_t s = 0, pB = Bp [j] ; pB < Bp [j+1] ;
                                pB++, s++)
                        {
                            // get B(k,j)
                            int64_t k = Bi [pB] ;
                            // flop count for just B(k,j)
                            int64_t fl = (Ap [k+1] - Ap [k]) ;
                            Bflops2 [s] = fl ;
                        }

                        // cumulative sum of flops to compute A*B(:,j)
                        GB_cumsum (Bflops2, bjnz, NULL, nthreads) ;

                        // slice B(:,j) into fine tasks
                        int nfine_this = ceil (jflops / target_task_size) ;
                        GB_pslice (&Fine_slice, Bflops2, bjnz, nfine_this) ;

                        // construct the fine tasks for B(:,j)
                        for (int fid = 0 ; fid < nfine_this ; fid++)
                        {
                            int64_t pstart = Fine_slice [fid] ;
                            int64_t pend   = Fine_slice [fid+1] ;
                            int64_t fl = Bflops2 [pend] - Bflops2 [pstart] ;
                            int64_t hsize = GB_hash_table_size (fl, cnrows) ;
                            TaskList [nf].start  = pB_start + pstart ;
                            TaskList [nf].end    = pB_start + pend - 1 ;
                            TaskList [nf].vector = j ;
                            TaskList [nf].hsize  = hsize ;
                            nf++ ;
                        }
                    }
                }

                // flush the last coarse task, if any
                if (jcoarse_start < j2)
                {
                    // jcoarse_start:j-1 form a single coarse task
                    GB_create_coarse_task (jcoarse_start, j2-1,
                        TaskList, nc++, Bflops, cnrows) ;
                }

            }
            else
            {
                // This coarse task is OK as-is.
                GB_create_coarse_task (j1, j2 - 1,
                    TaskList, nc++, Bflops, cnrows) ;
            }
        }

        // free workspace
        mxFree (Bflops2) ;          Bflops2 = NULL ;
        mxFree (Fine_slice) ;       Fine_slice = NULL ;
        mxFree (Coarse_initial) ;   Coarse_initial = NULL ;

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
        printf ("Fine %3d: ["GBd"] ("GBd" : "GBd") hsize/n %g\n",
            fid, j, p1, p2, ((double) hsize) / ((double) cnrows)) ;
        if (p1 > p2) printf (":::::::::::::::::: empty task\n") ;
        if (j < 0 || j > cnvec) mexErrMsgTxt ("j bad") ;
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
    }
#endif

    // Bflops is no longer needed as an alias for Cp
    Bflops = NULL ;

    #if GB_TIMING
    double t2 = simple_toc (tic) ;
    printf ("t2: task time %g\n", t2) ;
    simple_tic (tic) ;
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
    // #define GB_HASH_PAD (64 / (sizeof (double)))
    #define GB_HASH_PAD 0

    int64_t Hi_size_total = 1 ;
    int64_t Hx_size_total = 1 ;

    // determine the total size of all hash tables
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t hsize = TaskList [taskid].hsize ;
        if (hsize < cnrows)
        {
            Hi_size_total += (hsize + GB_HASH_PAD) ;
        }
        Hx_size_total += (hsize + GB_HASH_PAD) ;
    }

    // allocate space for all hash tables
    int64_t *restrict Hi_all = mxMalloc (Hi_size_total * sizeof (int64_t)) ;
    int64_t *restrict Hf_all = mxCalloc (Hx_size_total , sizeof (int64_t)) ;
    double  *restrict Hx_all = mxMalloc (Hx_size_total * sizeof (double )) ;

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
        TaskList [taskid].Hi = Hi ;
        TaskList [taskid].Hf = Hf ;
        TaskList [taskid].Hx = Hx ;

        int64_t hsize = TaskList [taskid].hsize ;
        if (hsize < cnrows)
        {
            Hi += (hsize + GB_HASH_PAD) ;
        }
        Hf += (hsize + GB_HASH_PAD) ;
        Hx += (hsize + GB_HASH_PAD) ;
    }

    //==========================================================================
    // symbolic phase: count # of entries in each vector of C or F
    //==========================================================================

    // Coarse tasks: compute nnz (C(:,j1:j2))
    // Fine tasks: compute nnz (F{taskid}) where F{taskid} = A*B(k1:k2,j).
    // For a vector j computed by fine tasks, nnz (C (:,j)) is not yet computed.

    #if GB_TIMING
    int nfine_hash = 0 ;
    int ncoarse_hash = 0 ;
    #endif

    // #pragma omp parallel for schedule(dynamic,1)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t *restrict Hi = TaskList [taskid].Hi ;
        int64_t *restrict Hf = TaskList [taskid].Hf ;
        double  *restrict Hx = TaskList [taskid].Hx ;
        int64_t j = TaskList [taskid].vector ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cnrows) ;
        bool is_fine = (j >= 0) ;

        //----------------------------------------------------------------------
        // do the task
        //----------------------------------------------------------------------

        if (is_fine)
        {

            //------------------------------------------------------------------
            // fine task: compute nnz (F{taskid}) where F{taskid} = A*B(k1:k2,j)
            //------------------------------------------------------------------

            int64_t pB_start = TaskList [taskid].start ;
            int64_t pB_end   = TaskList [taskid].end ;
            int64_t fjnz = 0 ;
            int64_t bjnz = pB_end - pB_start + 1 ;
            if (bjnz == 0)
            {

                // nothing to do

            }
            else if (bjnz == 1)
            {

                // F{taskid} = A(:,k)*B(k,j) for a single entry B(k,j)
                int64_t k = Bi [pB_start] ;
                fjnz = Ap [k+1] - Ap [k] ;

                #if GB_TIMING
                nquick++ ;
                #endif

            }
            else if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // Gustavson's method
                //--------------------------------------------------------------

                for (int64_t pB = pB_start ; pB <= pB_end ; pB++)
                {
                    // get B(k,j)
                    int64_t k = Bi [pB] ;
                    // scan A(:,k)
                    for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
                    {
                        // get A(i,k)
                        int64_t i = Ai [pA] ;
                        // add i to the gather/scatter workspace
                        if (Hf [i] != GB_FINE_MARK_1ST)
                        {
                            Hf [i] = GB_FINE_MARK_1ST ;
                            fjnz++ ;
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // hash method
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
                            if (Hf [hash] == GB_FINE_MARK_1ST)
                            {
                                // hash entry is occuppied
                                int64_t h = Hi [hash] ;
                                if (h == i)
                                {
                                    // i already in the hash table, at Hi [hash]
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
                                Hf [hash] = GB_FINE_MARK_1ST ;
                                Hi [hash] = i ;
                                fjnz++ ;
                                break ;
                            }
                        }
                    }
                }
            }

            TaskList [taskid].fjnz = fjnz ;

        }
        else
        {

            //------------------------------------------------------------------
            // coarse task: compute nnz in each A*B(:,j1:j2)
            //------------------------------------------------------------------

            int64_t j1 = TaskList [taskid].start ;
            int64_t j2 = TaskList [taskid].end ;
            int64_t mark = 0 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // Gustavson's method
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
                // hash method
                //--------------------------------------------------------------

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
    double t3 = simple_toc (tic) ;
    printf ("t3: sym1 time %g   (%d %d):%d\n", t3, ncoarse_hash, nfine_hash,
        nquick) ;
    simple_tic (tic) ;
    nquick = 0 ;
    #endif

    //==========================================================================
    // allocate F{taskid} for each fine task
    //==========================================================================

    // find the total size of all F{taskid}, for all fine tasks
    int64_t fnz = 0 ;
    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int64_t fjnz = TaskList [taskid].fjnz ;
        fnz += fjnz ;
    }

    // allocate space for F{taskid} for all fine tasks
    int64_t *restrict Fi_all = mxMalloc ((fnz + 1) * sizeof (double)) ;
    double  *restrict Fx_all = mxMalloc ((fnz + 1) * sizeof (double)) ;

    printf ("fnz %g flops %g  fnz/flops %g\n",
        (double) fnz, (double) total_flops,
        (double) fnz / (double) total_flops) ;

    if (0)
    {
        // out of memory
    }

    // split the space for each fine task
    int64_t *restrict Fi = Fi_all ;
    double  *restrict Fx = Fx_all ;

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int64_t fjnz = TaskList [taskid].fjnz ;
        TaskList [taskid].Fi = Fi ;
        TaskList [taskid].Fx = Fx ;
        Fi += fjnz ;
        Fx += fjnz ;
    }

    //==========================================================================
    // first numerical phase for fine tasks: compute F{taskid}
    //==========================================================================

    // Each fine task computes F{taskid} = A*B(k1:k2,j), both the pattern and
    // the values.  This step must be done before any coarse task can compute
    // C(:,j1:j2), since that requires all of Cp to first be found.  When
    // computing C, all tasks need to know where to place their results in the
    // final C matrix.

    #if GB_TIMING
    nfine_hash = 0 ;
    ncoarse_hash = 0 ;
    #endif

    // #pragma omp parallel for schedule(dynamic,1)
    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t *restrict Hi = TaskList [taskid].Hi ;
        int64_t *restrict Hf = TaskList [taskid].Hf ;
        double  *restrict Hx = TaskList [taskid].Hx ;

        int64_t j = TaskList [taskid].vector ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size == cnrows) ;

        int64_t pB_start = TaskList [taskid].start ;
        int64_t pB_end   = TaskList [taskid].end ;

        int64_t *restrict Fi = TaskList [taskid].Fi ;
        double  *restrict Fx = TaskList [taskid].Fx ;

        int64_t fjnz = TaskList [taskid].fjnz ;
        int64_t pF = 0 ;

        int64_t bjnz = pB_end - pB_start + 1 ;

        if (bjnz == 0)
        {

            // nothing to do

        }
        else if (bjnz == 1)
        {

            //------------------------------------------------------------------
            // F = A(:,k)*B(k,j) for a single entry B(k,j)
            //------------------------------------------------------------------

            // get B(k,j)
            int64_t k  = Bi [pB_start] ;
            double bkj = Bx [pB_start] ;
            // scan A(:,k)
            for (int64_t pA = Ap [k] ; pA < Ap [k+1] ; pA++)
            {
                // get A(i,k)
                int64_t i  = Ai [pA] ;
                double aik = Ax [pA] ;
                double t = aik * bkj ;
                // F{taskid}(i) += A(i,k)*B(k,j)
                Fx [pF] = t ;
                Fi [pF] = i ;
                pF++ ;
            }

            #if GB_TIMING
            nquick++ ;
            #endif

        }
        else if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // Gustavson's method
            //------------------------------------------------------------------

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
                    double t = aik * bkj ;
                    // F{taskid}(i) += A(i,k)*B(k,j)
                    if (Hf [i] != GB_FINE_MARK_2ND)
                    {
                        Hf [i] = GB_FINE_MARK_2ND ;
                        Hx [i] = t ;
                        Fi [pF++] = i ;
                    }
                    else
                    {
                        Hx [i] += t ;
                    }
                }
            }

            // gather the values into F (do not sort Fi)
            for (int64_t p = 0 ; p < fjnz ; p++)
            {
                int64_t i = Fi [p] ;
                Fx [p] = Hx [i] ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // hash method
            //------------------------------------------------------------------

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
                    double t = aik * bkj ;
                    // F{taskid}(i) += A(i,k)*B(k,j)
                    int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                    while (1)
                    {
                        if (Hf [hash] == GB_FINE_MARK_2ND)
                        {
                            // hash entry is occuppied
                            int64_t h = Hi [hash] ;
                            if (h == i)
                            {
                                // i already in the hash table, at Hi [hash]
                                Hx [hash] += t ;
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
                            Hf [hash] = GB_FINE_MARK_2ND ;
                            Hi [hash] = i ;
                            Hx [hash] = t ;
                            Fi [pF++] = i ;
                            break ;
                        }
                    }
                }
            }

            // gather the values into F (do not sort Fi)
            for (int64_t p = 0 ; p < fjnz ; p++)
            {
                int64_t i = Fi [p] ;
                // find F(i,j) in the hash table
                int64_t hash = (i * GB_HASH_FACTOR) & (hash_bits) ;
                while (1)
                {
                    int64_t h = Hi [hash] ;
                    if (h == i)
                    {
                        // i already in the hash table, at Hi [hash]
                        Fx [p] = Hx [hash] ;
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
    double t4 = simple_toc (tic) ;
    printf ("t4: fine1 time %g   (%d):%d\n", t4, nfine_hash, nquick) ;
    simple_tic (tic) ;
    nquick = 0 ;
    #endif

    //==========================================================================
    // merge hash tables for fine tasks
    //==========================================================================

    // Each vector C(:,j) computed by fine tasks is summed in a single merged
    // hash table, which consists of all hash tables used by those fine tasks.

    int64_t jlast = -1 ;

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {

        // this fine task operates on C(:,j)
        int64_t j = TaskList [taskid].vector ;
        if (j == jlast)
        {
            // this task is not the first fine task for vector j
            continue ;
        }
        jlast = j ;

        // clear the nnz (C (:,j)) count
        Cp [j] = 0 ;

        // find the range fine tasks that contribute to vector C(:,j)
        int taskid_start = taskid ;
        int taskid_end   = ntasks ;
        for (int tid = taskid ; tid < ntasks ; tid++)
        {
            if (j != TaskList [tid].vector)
            {
                taskid_end = tid ;
                break ;
            }
        }

        // merge all fine task hash tables for this vector j
        // int64_t *Hi = TaskList [taskid_start].Hi ;
        int64_t *Hf = TaskList [taskid_start].Hf ;
        double  *Hx = TaskList [taskid_start].Hx ;

        int64_t merge_hash_size = 0 ;
        for (int tid = taskid_start ; tid < taskid_end ; tid++)
        {
            int64_t hash_size = TaskList [tid].hsize ;
            merge_hash_size += hash_size ;
            // TaskList [tid].Hi = Hi ; (not needed)
            TaskList [tid].Hf = Hf ;
            TaskList [tid].Hx = Hx ;
        }

        for (int tid = taskid_start ; tid < taskid_end ; tid++)
        {
            TaskList [tid].hsize = merge_hash_size ;
        }
    }

// TODO: consider an alternative approach for fine tasks, which doesn't use any
// atomics:  Since Fi and Fx are single arrays, they can be merged instead of
// merging the hash tables.  Then, for each vector j computed by fine tasks, Fi
// and Fx can be sorted in parallel, and then duplicates summed (use
// GrB_builder), using the monoid operator.  Then Fi and Fx contain the vector
// C(:,j).

    //==========================================================================
    // symbolic phase: compute nnz (C (:,j)) for each set of fine tasks
    //==========================================================================

    // At this point, the hash table Hf contains only 3 unique entries:
    //
    //      0 (binary 00):  not modified, from the original calloc of Hf
    //      1 (binary 01):  first mark, GB_FINE_MARK_1ST
    //      2 (binary 10):  second mark, GB_FINE_MARK_2ND
    //
    // The upper 62 bits are all zero, in all cases.  In the following phase,
    // the lower order bits are set to 3 (11 in binary) to denote that the hash
    // entry is occuppied.  If the Hash method is used, the leading 62 bits are
    // used for the index i.  Since the all-zero case already appears,
    // (i+1) is held in the 62 bits, so it is in the range 1 to cnrows and
    // thus nonzero.

    #if GB_TIMING
    nfine_hash = 0 ;
    #endif

    // #pragma omp parallel for schedule(dynamic,1)
    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        // int64_t *restrict Hi = TaskList [taskid].Hi ; not used
        int64_t *restrict Hf = TaskList [taskid].Hf ;
        double  *restrict Hx = TaskList [taskid].Hx ;
        int64_t j = TaskList [taskid].vector ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size >= cnrows) ;
        int64_t *restrict Fi = TaskList [taskid].Fi ;
        double  *restrict Fx = TaskList [taskid].Fx ;
        int64_t fjnz = TaskList [taskid].fjnz ;
        int64_t my_cjnz = 0 ;

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // Gustavson's method
            //------------------------------------------------------------------

            for (int64_t pF = 0 ; pF < fjnz ; pF++)
            {
                // get F{taskid}(i)
                int64_t i = Fi [pF] ;
                // parallel: atomic swap
                int64_t v ;
                {
                    v = Hf [i] ; Hf [i] = GB_FINE_MARK_3RD ;
                }
                if (v != GB_FINE_MARK_3RD)
                {
                    my_cjnz++ ;             // # unique entries from this task
                    Fi [pF] = GB_FLIP (i) ; // mark this entry as unique
                    Hx [i] = 0 ;            // clear Hx for numeric phase
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // hash method
            //------------------------------------------------------------------

            // Only Hf is used for atomic updates.  If Hf [hash] is occuppied,
            // its 2 least significant bits are equal to 11, and the remaining
            // bits give the index i contained in that entry.

            #if GB_TIMING
            nfine_hash++ ;
            #endif

            for (int64_t pF = 0 ; pF < fjnz ; pF++)
            {
                // get F{taskid}(i)
                int64_t i = Fi [pF] ;
                // find i in the hash table
                int64_t hash = (i * GB_HASH_FACTOR) % (hash_size) ;
                while (1)
                {
                    // parallel: atomic read and modify
                    int64_t v ;
                    {
                        v = Hf [hash] ; Hf [hash] |= GB_FINE_MARK_3RD ;
                    }
                    if ((v & GB_FINE_MARK_3RD) == GB_FINE_MARK_3RD)
                    {
                        // hash entry is occuppied.  It might be in the process
                        // of being modified by the task that owns the entry.
                        // Spin-wait until the other tasks writes its value x,
                        // in the atomic write below.
                        while (v == GB_FINE_MARK_3RD)
                        {
                            // parallel: atomic read
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
                            hash = (hash + 1) % (hash_size) ;
                        }
                    }
                    else
                    {
                        // hash entry is not occuppied;
                        // add i to the hash table at this location
                        int64_t x = ((i+1) << 2) | GB_FINE_MARK_3RD ;
                        // parallel: atomic write
                        {
                            Hf [hash] = x ;
                        }
                        my_cjnz++ ;     // # unique entries found by this task
                        Fi [pF] = GB_FLIP (i) ; // mark this entry as unique
                        Hx [hash] = 0 ;         // clear Hx for numeric phase
                        break ;
                    }
                }
            }
        }

        TaskList [taskid].cp = my_cjnz ;
    }

    #if GB_TIMING
    double t5 = simple_toc (tic) ;
    printf ("t5: fine2 time %g    (%d)\n", t5, nfine_hash) ;
    simple_tic (tic) ;
    #endif

    //==========================================================================
    // cumulative sum of each fine task
    //==========================================================================

    // TaskList [taskid].cp is the # of unique entries found in C(:,j).  A
    // cumulative sum of these terms determines where each fine task can place
    // its unique entries in the final pattern of C(:,j).

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int64_t j = TaskList [taskid].vector ;
        int64_t my_cjnz = TaskList [taskid].cp ;
        TaskList [taskid].cp = Cp [j] ;
        Cp [j] += my_cjnz ;
    }

    // Cp [j] is now nnz (C (:,j)), for all vectors j, whether computed by fine
    // tasks or coarse tasks.

    //==========================================================================
    // compute Cp with cumulative sum
    //==========================================================================

    GB_cumsum (Cp, cncols, nonempty_result, 1 /* nthreads */) ;
    int64_t cnz = Cp [cncols] ;

    //==========================================================================
    // allocate Ci and Cx
    //==========================================================================

    int64_t *restrict Ci = mxMalloc (GB_IMAX (cnz, 1) * sizeof (int64_t)) ;
    double  *restrict Cx = mxMalloc (GB_IMAX (cnz, 1) * sizeof (double )) ;

    if (0)
    {
        // out of memory
    }

    #if GB_TIMING
    double t6 = simple_toc (tic) ;
    printf ("t6: cumsum time %g\n", t5) ;
    simple_tic (tic) ;
    #endif

    //==========================================================================
    // numeric phase
    //==========================================================================

    // Coarse tasks compute their results directly in C(:,j1:j2), with no
    // atomics needed.  Fine tasks assemble their results, F{taskid}, into
    // C(:,j) using atomics.

    #if GB_TIMING
    nfine_hash = 0 ;
    ncoarse_hash = 0 ;
    int64_t njd = 0, njd2 = 0 ;
    #endif

    // #pragma omp parallel for schedule(dynamic,1)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t *restrict Hi = TaskList [taskid].Hi ;
        int64_t *restrict Hf = TaskList [taskid].Hf ;
        double  *restrict Hx = TaskList [taskid].Hx ;
        int64_t j = TaskList [taskid].vector ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size >= cnrows) ;
        bool is_fine = (j >= 0) ;

        if (is_fine)
        {

            //------------------------------------------------------------------
            // fine task: compute C(:,j) via atomics
            //------------------------------------------------------------------

            int64_t *restrict Fi = TaskList [taskid].Fi ;
            double  *restrict Fx = TaskList [taskid].Fx ;
            int64_t fjnz = TaskList [taskid].fjnz ;

            int64_t pC = Cp [j] + TaskList [taskid].cp ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // Gustavson's method
                //--------------------------------------------------------------

                int64_t cjnz = Cp [j+1] - Cp [j] ;

                if (cjnz == cnrows || cjnz > cnrows / 16)
                {
                    // no need to save the pattern of C(:,j), since it will be
                    // either recreated (if C(:,j) is entirely dense) or
                    // created via an O(cnrows) scan of all of Hf.

                    for (int64_t pF = 0 ; pF < fjnz ; pF++)
                    {
                        // get F{taskid}(i)
                        int64_t i = Fi [pF] ;
                        if (GB_IS_FLIPPED (i))
                        {
                            // this entry is unique
                            i = GB_FLIP (i) ;
                        }
                        // parallel: atomic update
                        {
                            // C(i,j) += F{taskid}(i)
                            Hx [i] += Fx [pF] ;
                        }
                    }

                }
                else
                {

                    for (int64_t pF = 0 ; pF < fjnz ; pF++)
                    {
                        // get F{taskid}(i)
                        int64_t i = Fi [pF] ;
                        if (GB_IS_FLIPPED (i))
                        {
                            // this entry is unique; add it to Ci
                            i = GB_FLIP (i) ;
                            Ci [pC++] = i ;
                        }
                        // parallel: atomic update
                        {
                            // C(i,j) += F{taskid}(i)
                            Hx [i] += Fx [pF] ;
                        }
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // hash method
                //--------------------------------------------------------------

                #if GB_TIMING
                nfine_hash++ ;
                #endif

                for (int64_t pF = 0 ; pF < fjnz ; pF++)
                {
                    // get F{taskid}(i)
                    int64_t i = Fi [pF] ;
                    if (GB_IS_FLIPPED (i))
                    {
                        // this entry is unique; add it to Ci
                        i = GB_FLIP (i) ;
                        Ci [pC++] = i ;
                    }
                    // find i in the hash table
                    int64_t hash = (i * GB_HASH_FACTOR) % (hash_size) ;
                    while (1)
                    {
                        // parallel: atomic read
                        int64_t v ;
                        {
                            v = Hf [hash] ;
                        }
                        int64_t h = (v / 4) - 1 ;
                        if (h == i)
                        {
                            // i already in the hash table, at Hf [hash]
                            // parallel: atomic update
                            {
                                // C(i,j) += F{taskid}(i)
                                Hx [hash] += Fx [pF] ;
                            }
                            break ;
                        }
                        else
                        {
                            // linear probing
                            hash = (hash + 1) % (hash_size) ;
                        }
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // coarse task: compute C(:,j1:j2)
            //------------------------------------------------------------------

            int64_t j1 = TaskList [taskid].start ;
            int64_t j2 = TaskList [taskid].end ;
            int64_t mark = j2 - j1 + 2 ;

            if (use_Gustavson)
            {

                //--------------------------------------------------------------
                // Gustavson's method
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
                            double t = aik * bkj ;
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

                        for (int64_t p = pC, i = 0 ; p < Cp [j+1] ; p++, i++)
                        {
                            Ci [p] = i ;
                            Cx [p] = 0 ;
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
                                Cx [pC + i] += aik * bkj ;
                            }
                        }

                    }
                    else if (cjnz > cnrows / 16)
                    {

                        //------------------------------------------------------
                        // C(:,j) is not very sparse
                        //------------------------------------------------------

                        #if GB_TIMING
                        njd ++ ;
                        #endif

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
                                double t = aik * bkj ;
                                // update C(i,j) in gather/scatter workspace
                                if (Hf [i] != mark)
                                {
                                    // C(i,j) = A(i,k) * B(k,j)
                                    Hf [i] = mark ;
                                    Hx [i] = t ;
                                }
                                else
                                {
                                    // C(i,j) = A(i,k) * B(k,j)
                                    Hx [i] += t ;
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
                                double t = aik * bkj ;
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
                                    Hx [i] += t ;
                                }
                            }
                        }

                        // sort the pattern of C(:,j)
                        #if GB_TIMING
                        simple_tic (tic2) ;
                        #endif
                        GB_qsort_1a (Ci + Cp [j], cjnz) ;
                        #if GB_TIMING
                        tsort += simple_toc (tic2) ;
                        #endif

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
                // hash method
                //--------------------------------------------------------------

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
                            double t = aik * bkj ;
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
                                double t = aik * bkj ;
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
                                            Hx [hash] += t ;
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
                        #if GB_TIMING
                        simple_tic (tic2) ;
                        #endif
                        GB_qsort_1a (Ci + Cp [j], cjnz) ;
                        #if GB_TIMING
                        tsort += simple_toc (tic2) ;
                        #endif

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
    double t7 = simple_toc (tic) ;
    printf ("njd "GBd" "GBd"\n", njd, njd2) ;
    printf ("t7: num1 time %g   (%d %d):%d\n", t7, ncoarse_hash, nfine_hash,
        nquick) ;
    simple_tic (tic) ;
    nquick = 0 ;
    #endif

    //==========================================================================
    // final numeric phase: gather work for fine tasks
    //==========================================================================

    #if GB_TIMING
    nfine_hash = 0 ;
    #endif

    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t j = TaskList [taskid].vector ;
        if (taskid > 0 && j == TaskList [taskid-1].vector)
        {
            // only the first fine task does the gather
            continue ;
        }

        int64_t *restrict Hf = TaskList [taskid].Hf ;
        double  *restrict Hx = TaskList [taskid].Hx ;
        int64_t hash_size  = TaskList [taskid].hsize ;
        bool use_Gustavson = (hash_size >= cnrows) ;

        int64_t pC = Cp [j] ;
        int64_t cjnz = Cp [j+1] - pC ;

        //----------------------------------------------------------------------
        // gather the values into C(:,j)
        //----------------------------------------------------------------------

        if (use_Gustavson)
        {

            //------------------------------------------------------------------
            // Gustavson's method
            //------------------------------------------------------------------

            if (cjnz == cnrows)
            {

                //--------------------------------------------------------------
                // C(:,j) is entirely dense
                //--------------------------------------------------------------

                // #pragma omp parallel for schedule(static,1)
                for (int64_t p = pC, i = 0 ; p < Cp [j+1] ; p++, i++)
                {
                    Ci [p] = i ;
                }
                // TODO use GB_memcpy
                memcpy (Cx + pC, Hx, cnrows * sizeof (double)) ;

            }
            else if (cjnz > cnrows / 16)
            {

                //--------------------------------------------------------------
                // C(:,j) is not very sparse
                //--------------------------------------------------------------

                // O(cnrows) linear scan of Hf to create the pattern of C(:,j).
                // No sort is needed.

                // TODO do in parallel (2-pass method)
                for (int64_t i = 0 ; i < cnrows ; i++)
                {
                    if (Hf [i] == GB_FINE_MARK_3RD)
                    {
                        Ci [pC] = i ;
                        Cx [pC] = Hx [i] ;
                        pC++ ;
                    }
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C(:,j) is very sparse
                //--------------------------------------------------------------

                #if GB_TIMING
                simple_tic (tic2) ;
                #endif

                // TODO use a parallel sort
                GB_qsort_1a (Ci + Cp [j], cjnz) ;

                #if GB_TIMING
                tsort += simple_toc (tic2) ;
                #endif

                // TODO do in parallel
                for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
                {
                    // get C(i,j)
                    int64_t i = Ci [p] ;
                    Cx [p] = Hx [i] ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // hash method
            //------------------------------------------------------------------

            // sort the pattern of C(:,j)
            #if GB_TIMING
            simple_tic (tic2) ;
            #endif

            // TODO use a parallel sort
            GB_qsort_1a (Ci + Cp [j], cjnz) ;

            #if GB_TIMING
            tsort += simple_toc (tic2) ;
            nfine_hash++ ;
            #endif

            // TODO do in parallel
            for (int64_t p = Cp [j] ; p < Cp [j+1] ; p++)
            {
                // get C(i,j)
                int64_t i = Ci [p] ;
                // find i in the hash table
                int64_t hash = (i * GB_HASH_FACTOR) % (hash_size) ;
                while (1)
                {
                    // if parallel: atomic read
                    int64_t v ;
                    {
                        v = Hf [hash] ;
                    }
                    int64_t h = (v / 4) - 1 ;
                    if (h == i)
                    {
                        // i already in the hash table, at Hf [hash]
                        Cx [p] = Hx [hash] ;
                        break ;
                    }
                    else
                    {
                        // linear probing
                        hash = (hash + 1) % (hash_size) ;
                    }
                }
            }
        }
    }

    #if GB_TIMING
    double t8 = simple_toc (tic) ;
    printf ("t8: num2 time %g  (%d %d)\n", t8, ncoarse_hash, nfine_hash) ;

    double tot = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 ;
    printf ("   t1 %10.2f (compute flop counts)\n", 100 * t1/tot) ;
    printf ("   t2 %10.2f (create tasks)\n", 100 * t2/tot) ;
    printf ("   t3 %10.2f (sym1)\n", 100 * t3/tot) ;
    printf ("   t4 %10.2f (fine1)\n", 100 * t4/tot) ;
    printf ("   t5 %10.2f (fine2)\n", 100 * t5/tot) ;
    printf ("   t6 %10.2f (cumsum)\n", 100 * t6/tot) ;
    printf ("   t7 %10.2f (num1)\n", 100 * t7/tot) ;
    printf ("   t8 %10.2f (num2)\n", 100 * t8/tot) ;
    printf ("   total time %g\n", tot) ;
    printf ("   total flops %g\n", (double) total_flops) ;
    printf ("   total sort time %g sec,  tsort/tot: %10.2f\n",
        tsort, tsort / tot) ;
    #endif

    //==========================================================================
    // free workspace and return result
    //==========================================================================

    (*Cp_handle) = Cp ;
    (*Ci_handle) = Ci ;
    (*Cx_handle) = Cx ;

    mxFree (TaskList) ;

    mxFree (Fi_all) ;
    mxFree (Fx_all) ;

    mxFree (Hi_all) ;
    mxFree (Hx_all) ;
    mxFree (Hf_all) ;

    return (0) ;
}

