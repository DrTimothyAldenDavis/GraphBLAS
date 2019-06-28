//------------------------------------------------------------------------------
// GB_Pending_merge: merge pending tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Each GB_subassign_method* creates a set of Pending tuple objects, one per
// task.  After all tasks are finished, the pending tuples are merged into
// the single Pending object for the final matrix.

#include "GB_Pending.h"

bool GB_Pending_merge                   // merge pending tuples from each task
(
    GB_Pending *PHandle,                // input/output
    const GrB_Type type,
    const GrB_BinaryOp op,
    const bool is_matrix,
    const GB_task_struct *TaskList,     // list of subassign tasks
    const int ntasks,                   // number of tasks
    const int nthreads                  // number of threads
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (PHandle != NULL) ;
    ASSERT (TaskList != NULL) ;
    GB_Pending Pending = (*PHandle) ;

    //--------------------------------------------------------------------------
    // count the total number of new pending tuples
    //--------------------------------------------------------------------------

    int64_t Count [ntasks+1] ;
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        GB_Pending TaskPending = TaskList [taskid].Pending ;
        Count [taskid] = (TaskPending == NULL) ? 0 : TaskPending->n ;
        // if (Count [taskid] > 0) printf (" %d:"GBd, taskid, Count [taskid]) ;
    }

    GB_cumsum (Count, ntasks, NULL, 1) ;
    int64_t nnew = Count [ntasks] ;

    if (nnew == 0)
    {
        // quick return
        return (true) ;
    }
    
    // printf ("\n") ;

    //--------------------------------------------------------------------------
    // ensure the target list of Pending tuples exists
    //--------------------------------------------------------------------------

    if (Pending == NULL)
    {
        if (!GB_Pending_alloc (PHandle, type, op, is_matrix))
        {
            // out of memory
            return (false) ;
        }
        Pending = (*PHandle) ;
    }

    //--------------------------------------------------------------------------
    // ensure the target list of Pending tuples is large enough
    //--------------------------------------------------------------------------

    if (!GB_Pending_realloc (PHandle, nnew))
    {
        // out of memory
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // merge all lists
    //--------------------------------------------------------------------------

    int64_t nold = Pending->n ;
    size_t size = Pending->size ;
    int64_t *restrict Pending_i = Pending->i ;
    int64_t *restrict Pending_j = Pending->j ;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
        GB_Pending TaskPending = TaskList [taskid].Pending ;
        if (TaskPending != NULL)
        {
            int64_t n1 = nold + Count [taskid] ;
            int64_t n = TaskPending->n ;
            memcpy (Pending_i + n1, TaskPending->i, n * sizeof (int64_t)) ;
            if (Pending_j != NULL)
            {
                memcpy (Pending_j + n1, TaskPending->j, n * sizeof (int64_t)) ;
            }
            memcpy (Pending->x + (n1*size), TaskPending->x, n * size) ;
        }
    }

    Pending->n = nold + nnew ;

    //--------------------------------------------------------------------------
    // determine if the tuples from all tasks are sorted
    //--------------------------------------------------------------------------

    for (int taskid = 0 ; Pending->sorted && taskid < ntasks ; taskid++)
    {
        GB_Pending TaskPending = TaskList [taskid].Pending ;
        if (TaskPending != NULL)
        {
            Pending->sorted = Pending->sorted && TaskPending->sorted ;
            if (Pending->sorted)
            {
                int64_t n1 = nold + Count [taskid] ;
                if (n1 > 0)
                {
                    // (i,j) is the first pending tuple for this task; check
                    // with the pending tuple just before it in the merged
                    // list of pending tuples
                    int64_t i = Pending_i [n1] ;
                    int64_t j = (Pending_j != NULL) ? Pending_j [n1] : 0 ;
                    int64_t ilast = Pending_i [n1-1] ;
                    int64_t jlast = (Pending_j != NULL) ? Pending_j [n1-1] : 0 ;
                    Pending->sorted = (jlast < j) || (jlast == j && ilast <= i);
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (true) ;
}

