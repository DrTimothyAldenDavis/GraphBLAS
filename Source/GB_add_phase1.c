//------------------------------------------------------------------------------
// GB_add_phase1: find # of entries in C=A+B, C<M>=A+B, or C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_add_phase1 counts the number of entries in each vector of C, for C=A+B,
// C<M>=A+B, or C<!M>=A+B, and then does a cumulative sum to find Cp.
// GB_add_phase1 is preceded by GB_add_phase0, which finds the non-empty
// vectors of C.  This phase is done entirely in parallel.

// C, M, A, and B can be standard sparse or hypersparse, as determined by
// GB_add_phase0.  All cases of the mask M are handled: not present, present
// and not complemented, and present and complemented.

// GB_wait computes A=A+T where T is the matrix of the assembled pending
// tuples.  A and T are disjoint, so this function does not need to examine
// the pattern of A and T at all.  No mask is used in this case.

// Cp is either freed by phase2, or transplanted into C.

// PARALLEL: done

#include "GB.h"

GrB_Info GB_add_phase1                  // count nnz in each C(:,j)
(
    int64_t **Cp_handle,                // output of size Cnvec+1
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    const bool A_and_B_are_disjoint,    // if true, then A and B are disjoint

    // tasks from GB_add_phase0b
    GB_task_struct *restrict TaskList,      // array of structs
    const int ntasks,                       // # of tasks

    // analysis from GB_add_phase0
    const int64_t Cnvec,
    const int64_t *restrict Ch,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const bool Ch_is_Mh,                // if true, then Ch == M->h

    const GrB_Matrix M,                 // optional mask, may be NULL
    const bool Mask_comp,               // if true, then M is complemented
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cp_handle != NULL) ;
    ASSERT (Cnvec_nonempty != NULL) ;
    ASSERT_OK (GB_check (A, "A for add phase1", GB0)) ;
    ASSERT_OK (GB_check (B, "B for add phase1", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for add phase1", GB0)) ;
    ASSERT (A->vdim == B->vdim) ;

    int64_t *restrict Cp = NULL ;
    (*Cp_handle) = NULL ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;
    // TODO reduce nthreads for small problem (work: about O(anz+bnz+Cnvec))

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    GB_CALLOC_MEMORY (Cp, GB_IMAX (2, Cnvec+1), sizeof (int64_t)) ;
    if (Cp == NULL)
    { 
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector of C
    //--------------------------------------------------------------------------

    #define GB_PHASE_1_OF_2
    #include "GB_add_template.c"

    // TODO make this a function; use in GB_emult:

    //--------------------------------------------------------------------------
    // local cumulative sum of the fine tasks
    //--------------------------------------------------------------------------

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t k = TaskList [taskid].kfirst ;
        if (TaskList [taskid].klast == -1)
        {
            // Compute the sum of all fine tasks for vector k, in Cp [k].  Also
            // compute the cumulative sum of TaskList [taskid].pC, for the
            // tasks that work on vector k.  The first fine task uses pC = 0,
            // which becomes an offset from the final Cp [k].  A subsequent
            // fine task t for a vector k starts on offset of TaskList [t].pC.
            // from the start of C(:,k).  Cp [k] has not been cumsum'd across
            // all of Cp.  It is just the count of the entries in C(:,k).  The
            // final Cp [k] is added to each fine task below, after the
            // GB_cumsum of Cp.
            int64_t pC = Cp [k] ;
            Cp [k] += TaskList [taskid].pC ;
            TaskList [taskid].pC = pC ;
        }
    }

    //--------------------------------------------------------------------------
    // replace Cp with its cumulative sum
    //--------------------------------------------------------------------------

    GB_cumsum (Cp, Cnvec, Cnvec_nonempty, nthreads) ;

    //--------------------------------------------------------------------------
    // shift the cumulative sum of the fine tasks
    //--------------------------------------------------------------------------

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        int64_t k = TaskList [taskid].kfirst ;
        if (TaskList [taskid].klast == -1)
        {
            // TaskList [taskid].pC is currently an offset for this task into
            // C(:,k).  The first fine task for vector k has an offset of zero,
            // the 2nd fine task has an offset equal to the # of entries
            // computed by the first task, and so on.  Cp [k] needs to be added
            // to all offsets to get the final starting position for each fine
            // task in C.
            TaskList [taskid].pC += Cp [k] ;
        }
        else
        {
            // The last fine task to operate on vector k needs know its own
            // pC_end, which is Cp [k+1].  Suppose that task is taskid-1.  If
            // this taskid is the first fine task for vector k, then TaskList
            // [taskid].pC is set to Cp [k] above.  If all coarse tasks are
            // also given TaskList [taskid].pC = Cp [k], then taskid-1 will
            // always know its pC_end, which is TaskList [taskid].pC.
            TaskList [taskid].pC = Cp [k] ;
        }
    }

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    (*Cp_handle) = Cp ;
    return (GrB_SUCCESS) ;
}

