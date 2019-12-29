//------------------------------------------------------------------------------
// GB_AxB_saxpy3.h: definitions for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_sort.h"
#ifndef GBCOMPACT
#include "GB_AxB__include.h"
#endif

// ceiling of a/b
#define GB_CEIL(a,b) (((a) + (b) - 1) / (b))
#define GB_NTASKS_PER_THREAD 2
#define GB_HASH_FACTOR 107
#define GB_TIMING 1
#define GB_COSTLY 1.2
#define GB_FINE_WORK 2

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
    int64_t *Hi ;       // Hi array for hash table (coarse hash tasks only)
    int64_t *Hf ;       // Hf array for hash table
    double  *Hx ;       // Hx array for hash table
    int64_t my_cjnz ;   // # entries in C(:,j) found by this fine task
    int64_t flops ;     // # of flops in this task
    int master ;        // master fine task for the vector C(:,j)
    int nfine_team_size ;   // # of fine tasks for vector C(:,j)
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
    int64_t j1,         // coarse task consists of vectors j1:j2, inclusive
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
    TaskList [taskid].hsize  = GB_hash_table_size (flmax, cnrows) ;
    TaskList [taskid].flops  = Bflops [j2+1] - Bflops [j1] ;
}

