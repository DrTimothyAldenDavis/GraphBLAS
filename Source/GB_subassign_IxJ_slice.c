//------------------------------------------------------------------------------
// GB_subassign_IxJ_slice: slice IxJ for subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Construct a set of tasks to compute C(I,J)<...> = x or += x, for a subassign
// method that performs scalar assignment, based on slicing the Cartesian
// product IxJ.  If enough tasks can be constructed by just slicing J, then all
// tasks are coarse.  Each coarse tasks computes all of C(I,J(kfirst:klast-1)),
// for its range of indices kfirst:klast-1, inclusive.

// Otherwise, if not enough coarse tasks can be constructed, then all tasks are
// fine.  Each fine task computes a slice of C(I(iA_start:iA_end-1), jC) for a
// single index jC = J(kfirst).

// This method is used by GB_subassign_methods 3, 4, 7, 8, 11a, 11b, 12a, 12b,
// which are the 8 scalar assignment methods that must iterate over all IxJ.

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   -   -   S        7: C(I,J) = x, with S
        //  -   -   -   +   -   -        3: C(I,J) += x
        //  -   -   -   +   -   S        8: C(I,J) += x, with S
        //  M   c   -   -   -   S      11b: C(I,J)<!M> = x, with S
        //  M   c   -   +   -   -        4: C(I,J)<!M> += x
        //  M   c   -   +   -   S      12b: C(I,J)<!M> += x, with S
        //  M   c   r   -   -   S      11a: C(I,J)<!M,repl> = x, with S
        //  M   c   r   +   -   S      12a: C(I,J)<!M,repl> += x, with S

// If fine tasks are constructed for Methods 3 and 4, each fine task must also
// slice its vector C(:,jC), and save the results in TaskList [ ].pC and
// pC_end, since the binary search cannot be done once C(:,jC) is being
// modified by multiple fine tasks.  The other methods do not need this search,
// since they all rely on the matrix S instead, which is not modified during
// the subassignment.

// There are 12 methods that perform scalar assignment: the 8 listed above, and
// Methods 1, 2, 11c, 12c.  The latter 4 methods do do not need to iterate over
// the entire IxJ space, because of the mask M:

        //  M   -   -   -   -   -        1: C(I,J)<M> = x
        //  M   -   -   +   -   -        2: C(I,J)<M> += x
        //  M   -   r   -   -   S      11c: C(I,J)<M,repl> = x, with S
        //  M   -   r   +   -   S      12c: C(I,J)<M,repl> += x, with S

// As a result, they do not use GB_subassign_IxJ_slice to define their tasks.
// Instead, Methods 1 and 2 slice the matrix M, and Methods 11c and 12c slice
// the matrix addition M+S.

#include "GB_subassign.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                                     \
{                                                                       \
    GB_FREE_MEMORY (TaskList, max_ntasks+1, sizeof (GB_task_struct)) ;  \
}

//------------------------------------------------------------------------------
// GB_subassign_IxJ_slice
//------------------------------------------------------------------------------

GrB_Info GB_subassign_IxJ_slice
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs, of size max_ntasks
    int *p_max_ntasks,              // size of TaskList
    int *p_ntasks,                  // # of tasks constructed
    int *p_nthreads,                // # of threads to use
    // input:
    const GrB_Matrix C,             // output matrix C (method 3 and 4 only)
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_TaskList != NULL) ;
    ASSERT (p_max_ntasks != NULL) ;
    ASSERT (p_ntasks != NULL) ;
    ASSERT (p_nthreads != NULL) ;
    ASSERT_OK_OR_NULL (GB_check (C, "C for IxJ_slice", GB0)) ;

    (*p_TaskList  ) = NULL ;
    (*p_max_ntasks) = 0 ;
    (*p_ntasks    ) = 0 ;
    (*p_nthreads  ) = 1 ;
    int ntasks, max_ntasks = 0, nthreads ;
    GB_task_struct *TaskList = NULL ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // get C, if present, for Methods 3 and 4
    //--------------------------------------------------------------------------

    const int64_t *restrict Cp = NULL ;
    const int64_t *restrict Ch = NULL ;
    const int64_t *restrict Ci = NULL ;
    bool C_is_hyper = false ;
    int64_t nzombies = 0 ;
    int64_t Cnvec = 0 ;
    int64_t cvlen = 0 ;
    if (C != NULL)
    { 
        Cp = C->p ;
        Ch = C->h ;
        Ci = C->i ;
        C_is_hyper = C->is_hyper ;
        nzombies = C->nzombies ;
        Cnvec = C->nvec ;
        cvlen = C->vlen ;
    }

    // printf ("nI "GBd" Ikind "GBd" Icolon "GBd" "GBd" "GBd"\n", nI, Ikind,
    //     Icolon [0], Icolon [1], Icolon [2]) ;
    // for (int64_t iA = 0 ; iA < nI ; iA++)
    // {
    //     int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
    //     printf ("   iA "GBd" iC "GBd"\n", iA, iC) ;
    // }

    // printf ("nJ "GBd" Jkind "GBd" Jcolon "GBd" "GBd" "GBd"\n", nJ, Jkind,
    //     Jcolon [0], Jcolon [1], Jcolon [2]) ;
    // for (int64_t jA = 0 ; jA < nJ ; jA++)
    // {
    //     int64_t jC = GB_ijlist (J, jA, Jkind, Jcolon) ;
    //     printf ("   jA "GBd" jC "GBd"\n", jA, jC) ;
    // }

    //--------------------------------------------------------------------------
    // allocate the initial TaskList
    //--------------------------------------------------------------------------

    double work = ((double) nI) * ((double) nJ) ;
    nthreads = GB_nthreads (work, chunk, nthreads_max) ;
    int ntasks0 = (nthreads == 1) ? 1 : (32 * nthreads) ;
    GB_REALLOC_TASK_LIST (TaskList, ntasks0, max_ntasks) ;

    //--------------------------------------------------------------------------
    // check for quick return for a single task
    //--------------------------------------------------------------------------

    if (nJ == 0 || ntasks0 == 1)
    { 
        // construct a single coarse task that does all the work
        TaskList [0].kfirst = 0 ;
        TaskList [0].klast  = nJ-1 ;
        (*p_TaskList  ) = TaskList ;
        (*p_max_ntasks) = max_ntasks ;
        (*p_ntasks    ) = (nJ == 0) ? 0 : 1 ;
        (*p_nthreads  ) = 1 ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // construct the tasks: all fine or all coarse
    //--------------------------------------------------------------------------

    // The desired number of tasks is ntasks0.  If this is less than or equal
    // to |J|, then all tasks can be coarse, and each coarse task handles one
    // or more indices in J.  Otherise, multiple fine tasks are constructed for
    // each index in J.

    if (ntasks0 <= nJ)
    {

        //----------------------------------------------------------------------
        // all coarse tasks: slice just J
        //----------------------------------------------------------------------

        ntasks = ntasks0 ;
        for (int taskid = 0 ; taskid < ntasks ; taskid++)
        {
            // the coarse task computes C (I, J (j:jlast-1))
            int64_t j, jlast ;
            GB_PARTITION (j, jlast, nJ, taskid, ntasks) ;
            ASSERT (j <= jlast) ;
            ASSERT (jlast <= nJ) ;
            TaskList [taskid].kfirst = j ;
            TaskList [taskid].klast  = jlast - 1 ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // all fine tasks: slice both I and J
        //----------------------------------------------------------------------

        // create at least 2 fine tasks per index in J
        int nI_fine_tasks = ntasks0 / nJ ;
        nI_fine_tasks = GB_IMAX (nI_fine_tasks, 2) ;
        ntasks = 0 ;

        GB_REALLOC_TASK_LIST (TaskList, nJ * nI_fine_tasks, max_ntasks) ;

        if (C == NULL)
        {

            //------------------------------------------------------------------
            // construct fine tasks for index j; no binary search of C
            //------------------------------------------------------------------

            // Method 7, 8, 11a, 11b, 12a, 12b: no need for binary search of C

            for (int64_t j = 0 ; j < nJ ; j++)
            {
                // create nI_fine_tasks for each index in J
                for (int t = 0 ; t < nI_fine_tasks ; t++)
                { 
                    // this fine task computes C (I (iA_start:iA_end-1), jC)
                    int64_t iA_start, iA_end ;
                    GB_PARTITION (iA_start, iA_end, nI, t, nI_fine_tasks) ;
                    TaskList [ntasks].kfirst = j ;
                    TaskList [ntasks].klast  = -1 ;
                    TaskList [ntasks].pA     = iA_start ;
                    TaskList [ntasks].pA_end = iA_end ;
                    ntasks++ ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // construct fine tasks for index j and perform binary search of C
            //------------------------------------------------------------------

            // Method 3 and 4: binary search in C(:,jC)

            for (int64_t j = 0 ; j < nJ ; j++)
            {
                // get the C(:,jC) vector where jC = J [j]
                int64_t GB_jC_LOOKUP ;
                bool jC_dense = ((pC_end - pC_start) == cvlen) ;

                // create nI_fine_tasks for each index in J
                for (int t = 0 ; t < nI_fine_tasks ; t++)
                {
                    // this fine task computes C (I (iA_start:iA_end-1), jC)
                    int64_t iA_start, iA_end ;
                    GB_PARTITION (iA_start, iA_end, nI, t, nI_fine_tasks) ;
                    TaskList [ntasks].kfirst = j ;
                    TaskList [ntasks].klast  = -1 ;
                    TaskList [ntasks].pA     = iA_start ;
                    TaskList [ntasks].pA_end = iA_end ;

                    if (jC_dense)
                    { 
                        // do not slice C(:,jC) if it is dense
                        TaskList [ntasks].pC     = pC_start ;
                        TaskList [ntasks].pC_end = pC_end ;
                    }
                    else
                    { 

                        // find where this task starts and ends in C(:,jC)
                        int64_t iC1 = GB_ijlist (I, iA_start, Ikind, Icolon) ;
                        int64_t iC2 = GB_ijlist (I, iA_end, Ikind, Icolon) ;

                        // If I is an explicit list, it must be already sorted
                        // in ascending order, and thus iC1 <= iC2.  If I is
                        // GB_ALL or GB_STRIDE with inc >= 0, then iC1 < iC2.
                        // But if inc < 0, then iC1 > iC2.  iC_start and iC_end
                        // are used for a binary search bracket, so iC_start <=
                        // iC_end must hold.
                        int64_t iC_start = GB_IMIN (iC1, iC2) ;
                        int64_t iC_end   = GB_IMAX (iC1, iC2) ;

                        // printf ("\niA_start "GBd"\n", iA_start) ;
                        // printf ("iA_end   "GBd"\n", iA_end) ;

                        // printf ("\niC_start "GBd"\n", iC_start) ;
                        // printf ("iC_end   "GBd"\n", iC_end) ;

                        // this task works on Ci,Cx [pC:pC_end-1]
                        int64_t pleft = pC_start ;
                        int64_t pright = pC_end - 1 ;
                        bool found, is_zombie ;
                        GB_BINARY_SPLIT_ZOMBIE (iC_start, Ci, pleft, pright,
                            found, nzombies, is_zombie) ;
                        TaskList [ntasks].pC = pleft ;

                        pleft = pC_start ;
                        pright = pC_end - 1 ;
                        GB_BINARY_SPLIT_ZOMBIE (iC_end, Ci, pleft, pright,
                            found, nzombies, is_zombie) ;
                        TaskList [ntasks].pC_end = (found) ? (pleft+1) : pleft ;
                    }

                    ASSERT (TaskList [ntasks].pC <= TaskList [ntasks].pC_end) ;
                    ntasks++ ;
                }
            }
        }
    }

    ASSERT (ntasks <= max_ntasks) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_TaskList  ) = TaskList ;
    (*p_max_ntasks) = max_ntasks ;
    (*p_ntasks    ) = ntasks ;
    (*p_nthreads  ) = nthreads ;
    return (GrB_SUCCESS) ;
}

