//------------------------------------------------------------------------------
// GB_AxB_saxpy3_cumsum: cumulative sum of Cp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// phase4: cumulative sum of C->p

// TODO: if C is a single vector computed via fine hash tasks only, skip this
// step.  Instead, allocate C->x and C->i as the upper bound (same as the hash
// table size).  Use atomic increment to grab a slot to place a single entry in
// C->i in phase 2.  Then in phase 5, iterate across C->i and gather from the
// hash table.  For the ANY monoid, phase 5 can be skipped, if the first
// value is placed in C->x in phase 2.

#include "GB_AxB_saxpy3.h"

int64_t GB_AxB_saxpy3_cumsum    // return cjnz_max for fine tasks
(
    GrB_Matrix C,               // finalize C->p
    GB_saxpy3task_struct *TaskList, // list of tasks, and workspace
    int nfine,                  // number of fine tasks
    double chunk,               // chunk size
    int nthreads                // number of threads
)
{

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Cp = C->p ;        // ok: C is sparse
    const int64_t cvlen = C->vlen ;
    const int64_t cnvec = C->nvec ;

    //==========================================================================
    // phase4: compute Cp with cumulative sum
    //==========================================================================

    //--------------------------------------------------------------------------
    // sum nnz (C (:,j)) for fine tasks
    //--------------------------------------------------------------------------

    // TaskList [taskid].my_cjnz is the # of unique entries found in C(:,j) by
    // that task.  Sum these terms to compute total # of entries in C(:,j).

    int taskid ;
    for (taskid = 0 ; taskid < nfine ; taskid++)
    { 
        int64_t kk = TaskList [taskid].vector ;
        Cp [kk] = 0 ;       // ok: C is sparse
    }

    for (taskid = 0 ; taskid < nfine ; taskid++)
    { 
        int64_t kk = TaskList [taskid].vector ;
        int64_t my_cjnz = TaskList [taskid].my_cjnz ;
        Cp [kk] += my_cjnz ;        // ok: C is sparse
        ASSERT (my_cjnz <= cvlen) ;
    }

    //--------------------------------------------------------------------------
    // cumulative sum for Cp (fine and coarse tasks)
    //--------------------------------------------------------------------------

    // Cp [kk] is now nnz (C (:,j)), for all vectors j, whether computed by
    // fine tasks or coarse tasks, and where j == GBH (Bh, kk) 

    int nth = GB_nthreads (cnvec, chunk, nthreads) ;
    GB_cumsum (Cp, cnvec, &(C->nvec_nonempty), nth) ;

    //--------------------------------------------------------------------------
    // cumulative sum of nnz (C (:,j)) for each team of fine tasks
    //--------------------------------------------------------------------------

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
                int64_t cjnz = Cp [kk+1] - Cp [kk] ;        // ok: C is sparse
                cjnz_max = GB_IMAX (cjnz_max, cjnz) ;
            }
        }
        int64_t my_cjnz = TaskList [taskid].my_cjnz ;
        TaskList [taskid].my_cjnz = cjnz_sum ;
        cjnz_sum += my_cjnz ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (cjnz_max) ;
}

