//------------------------------------------------------------------------------
// GB_saxpy3.h: definitions for C=A*B saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_AxB_saxpy3 method uses a mix of Gustavson's method and the Hash method,
// combining the two for any given C=A*B computation.

#ifndef GB_SAXPY3_H
#define GB_SAXPY3_H
#include "GB.h"

//------------------------------------------------------------------------------
// scalar used in the hash function
//------------------------------------------------------------------------------

#define GB_HASH_FACTOR 107

//------------------------------------------------------------------------------
// GB_saxpy3task_struct: task descriptor for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// A coarse task computes C(:,j1:j2) = A*B(:,j1:j2), for a contiguous set of
// vectors j1:j2.  A coarse taskid is denoted byTaskList [taskid].vector == -1,
// kfirst = TaskList [taskid].start, and klast = TaskList [taskid].end, and
// where j1 = (Bh == NULL) ? kstart : Bh [kstart] and likewise for j2.  No
// summation is needed for the final result of each coarse task.

// A fine taskid computes A*B(k1:k2,j) for a single vector C(:,j), for a
// contiguous range k1:k2, where kk = Tasklist[taskid].vector (which is >= 0),
// k1 = Bi [TaskList [taskid].start], k2 = Bi [TaskList [taskid].end].  It sums
// its computations in a hash table shared by all fine tasks that compute
// C(:,j), via atomics.  The vector index j is either kk if B is standard, or j
// = B->h [kk] if B is hypersparse.  The algorithn never needs to know the
// vector index j, however.

// Both tasks use a hash table allocated uniquely for the task, in Hi, Hf, and
// Hx.  The size of the hash table is determined by the maximum # of flops
// needed to compute any vector in C(:,j1:j2) for a coarse task, or the entire
// computation of the single vector in a fine task.  For the Hash method, the
// table has a size that is twice the smallest a power of 2 larger than the
// flop count.  If this size is a significant fraction of cnrows, then the Hash
// method is not used, and Gustavson's method is used, with the hash size is
// set to cnrows.

typedef struct
{
    int64_t start ;     // starting vector for coarse task, p for fine task
    int64_t end ;       // ending vector for coarse task, p for fine task
    int64_t vector ;    // -1 for coarse task, vector j for fine task
    int64_t hsize ;     // size of hash table
    int64_t *Hi ;       // Hi array for hash table (coarse hash tasks only)
    GB_void *Hf ;       // Hf array for hash table (uint8_t or int64_t)
    GB_void *Hx ;       // Hx array for hash table
    int64_t my_cjnz ;   // # entries in C(:,j) found by this fine task
    int64_t flops ;     // # of flops in this task
    int master ;        // master fine task for the vector C(:,j)
    int nfine_team_size ;   // # of fine tasks for vector C(:,j)
}
GB_saxpy3task_struct ;

#endif

