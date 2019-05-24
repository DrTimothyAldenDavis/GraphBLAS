//------------------------------------------------------------------------------
// GB_ewise_slice: slice the entries and vectors for an ewise operation
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Constructs a set of tasks to compute C, for an element-wise operation
// (GB_add, GB_emult, and GB_mask) that operates on two input matrices,
// C=op(A,B).

#include "GB.h"

//------------------------------------------------------------------------------
// GB_allocate_task_list:  allocate a task list
//------------------------------------------------------------------------------

static inline GB_task_struct *GB_allocate_task_list
(
    int max_ntasks
)
{
    GB_task_struct *TaskList ;
    GB_MALLOC_MEMORY (TaskList, max_ntasks+1, sizeof (GB_task_struct)) ;
    if (TaskList != NULL)
    {
        for (int t = 0 ; t <= max_ntasks ; t++)
        {
            TaskList [t].kfirst = -1 ;
            TaskList [t].klast  = -1 ;
            TaskList [t].pA     = -1 ;
            TaskList [t].pB     = -1 ;
            TaskList [t].pC     =  0 ;
        }
    }
    return (TaskList) ;
}

//------------------------------------------------------------------------------
// GB_ewise_slice
//------------------------------------------------------------------------------

GrB_Info GB_ewise_slice
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs, of size max_ntasks
    int *p_max_ntasks,              // size of TaskList
    int *p_ntasks,                  // # of tasks constructed
    // input:
    const int64_t Cnvec,            // # of vectors of C
    const int64_t *restrict Ch,     // vectors of C, if hypersparse
    const int64_t *restrict C_to_M, // mapping of C to M
    const int64_t *restrict C_to_A, // mapping of C to A
    const int64_t *restrict C_to_B, // mapping of C to B
    const GrB_Matrix M,             // optional mask
    const bool Mask_comp,
    const GrB_Matrix A,             // matrix to slice
    const GrB_Matrix B,             // matrix to slice
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_TaskList != NULL) ;
    ASSERT (p_max_ntasks != NULL) ;
    ASSERT (p_ntasks != NULL) ;
    ASSERT_OK (GB_check (A, "A for ewise_slice", GB0)) ;
    ASSERT_OK (GB_check (B, "B for ewise_slice", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for ewise_slice", GB0)) ;

    (*p_TaskList  ) = NULL ;
    (*p_max_ntasks) = 0 ;
    (*p_ntasks    ) = 0 ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;
    // TODO reduce nthreads for small problems

    //--------------------------------------------------------------------------
    // quick return if only one task
    //--------------------------------------------------------------------------

// printf ("nthreads %d Cnvec "GBd"\n", nthreads, Cnvec) ;

    if (Cnvec == 0 || nthreads == 1)
    {
        GB_task_struct *TaskList = GB_allocate_task_list (1) ;
        if (TaskList == NULL)
        {
            // out of memory
            return (GB_OUT_OF_MEMORY) ;
        }
        // construct a single coarse task that computes all of C
        TaskList [0].kfirst = 0  ;
        TaskList [0].klast  = Cnvec-1  ;
        (*p_TaskList  ) = TaskList ;
        (*p_max_ntasks) = 1 ;
        (*p_ntasks    ) = 1 ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // get M, A, and B
    //--------------------------------------------------------------------------

    const int64_t vlen = A->vlen ;
    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ai = A->i ;
    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bi = B->i ;
    const int64_t *restrict Mp = (M == NULL) ? NULL : M->p ;
    const int64_t *restrict Mh = (M == NULL) ? NULL : M->h ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    int64_t *restrict Cwork ;
    GB_MALLOC_MEMORY (Cwork, Cnvec+1, sizeof (int64_t)) ;
    if (Cwork == NULL)
    {
        // out of memory
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // compute the work for each vector of C
    //--------------------------------------------------------------------------

    int nth = GB_nthreads (Cnvec, 4096, nthreads) ;
    #pragma omp parallel for num_threads(nth)
    for (int64_t k = 0 ; k < Cnvec ; k++)
    {
        int64_t j = (Ch == NULL) ? k : Ch [k] ;
        int64_t kA = (C_to_A == NULL) ? j : C_to_A [k] ;
        int64_t kB = (C_to_B == NULL) ? j : C_to_B [k] ;
        int64_t aknz = (kA < 0) ? 0 : (Ap [kA+1] - Ap [kA]) ;
        int64_t bknz = (kB < 0) ? 0 : (Bp [kB+1] - Bp [kB]) ;
        int64_t ckwork = aknz + bknz + 1 ;
        // printf ("aknz "GBd"\n", aknz) ;
        // printf ("bknz "GBd"\n", bknz) ;
        if (M != NULL && !Mask_comp)
        {
            int64_t kM = ((Ch == Mh) ? k :
                         ((C_to_M == NULL) ? j : C_to_M [k])) ;
            int64_t mknz = (kM >= 0) ? 0 : (Mp [kM+1] - Mp [kM]) ;
            if (mknz == 0)
            {
                // M(:,j) is empty, and the mask is not complemented.  Thus,
                // C(:,j) will be empty, and this is found in O(1) time.
                ckwork = 1 ;
            }
            else
            {
                // C(:,j) will have at most mknz entries, but this may require
                // that all of M(:,j), A(:,j), and B(:,j) be scanned.  So add
                // mknz to the total work.
                ckwork += mknz ;
            }
        }
        Cwork [k] = ckwork ;
    }

    //--------------------------------------------------------------------------
    // replace Cwork with its cumulative sum
    //--------------------------------------------------------------------------

//  for (int64_t k = 0 ; k < Cnvec ; k++)
//  {
//      printf ("Cwork ["GBd"] = "GBd"\n", k, Cwork [k]) ;
//  }

    GB_cumsum (Cwork, Cnvec, NULL, nthreads) ;
    double cwork = (double) Cwork [Cnvec] ;

//  printf ("\nafter cumsum:\n") ;
//  for (int64_t k = 0 ; k <= Cnvec ; k++)
//  {
//      printf ("Cwork ["GBd"] = "GBd"\n", k, Cwork [k]) ;
//  }

    //--------------------------------------------------------------------------
    // determine the number of tasks to create
    //--------------------------------------------------------------------------

    double target_task_size = cwork / (double) (32 * nthreads) ;
    target_task_size = GB_IMAX (target_task_size, 4096) ;
    int ntasks1 = cwork / target_task_size ;
    ntasks1 = GB_IMAX (ntasks1, 1) ;
//  printf ("ntasks1 %d\n", ntasks1) ;

    //--------------------------------------------------------------------------
    // allocate the task descriptions
    //--------------------------------------------------------------------------

    int max_ntasks = 2 * ntasks1 ;
    GB_task_struct *TaskList = GB_allocate_task_list (max_ntasks) ;
    if (TaskList == NULL)
    {
        // out of memory
        GB_FREE_MEMORY (Cwork, Cnvec+1, sizeof (int64_t)) ;
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // slice the work into coarse tasks
    //--------------------------------------------------------------------------

    // also see GB_pslice
    int Coarse [ntasks1+1] ;
    Coarse [0] = 0 ;
    int64_t k = 0 ;
    for (int t = 1 ; t < ntasks1 ; t++)
    {
        // find k so that Cwork [k] == t * target_task_size
        int64_t work = t * target_task_size ;
        int64_t pright = Cnvec ;
        GB_BINARY_TRIM_SEARCH (work, Cwork, k, pright) ;
        Coarse [t] = k ;
        // printf ("Coarse [%d] = "GBd"\n", t, Coarse [t]) ;
    }
    Coarse [ntasks1] = Cnvec ;
    // printf ("last Coarse [%d] = "GBd"\n", ntasks1, Coarse [ntasks1]) ;

    //--------------------------------------------------------------------------
    // construct all tasks, both coarse and fine
    //--------------------------------------------------------------------------

    int ntasks = 0 ;

    for (int t = 0 ; t < ntasks1 ; t++)
    {

        //----------------------------------------------------------------------
        // coarse task computes C (:,k:klast)
        //----------------------------------------------------------------------

        int64_t k = Coarse [t] ;
        int64_t klast  = Coarse [t+1] - 1 ;

        if (k >= Cnvec)
        {

            //------------------------------------------------------------------
            // all tasks have been constructed
            //------------------------------------------------------------------

            break ;

        }
        else if (k < klast)
        {

            //------------------------------------------------------------------
            // coarse task has 2 or more vectors
            //------------------------------------------------------------------

            // This is a non-empty coarse-grain task that does two or more
            // entire vectors of C, vectors k:klast, inclusive.
            ASSERT (ntasks < max_ntasks) ;
            TaskList [ntasks].kfirst = k ;
            TaskList [ntasks].klast  = klast ;
            ntasks++ ;

        }
        else
        {

            //------------------------------------------------------------------
            // coarse task has 0 or 1 vectors
            //------------------------------------------------------------------

            // As a coarse-grain task, this task is empty or does a single
            // vector, k.  Vector k must be removed from the work done by this
            // and any other coarse-grain task, and split into one or more
            // fine-grain tasks.

            for (int tt = t ; tt < ntasks1 ; tt++)
            {
                // remove k from the initial slice tt
                if (Coarse [tt] == k)
                {
                    // remove k from task tt
                    Coarse [tt] = k+1 ;
                }
                else
                {
                    break ;
                }
            }

            //------------------------------------------------------------------
            // determine the # of fine-grain tasks to create for vector k
            //------------------------------------------------------------------

            int64_t j = (Ch == NULL) ? k : Ch [k] ;
            int64_t kA = (C_to_A == NULL) ? j : C_to_A [k] ;
            int64_t kB = (C_to_B == NULL) ? j : C_to_B [k] ;

            int64_t pA_start = (kA < 0) ? -1 : Ap [kA] ;
            int64_t pA_end   = (kA < 0) ? -1 : Ap [kA+1] ;
            int64_t pB_start = (kB < 0) ? -1 : Bp [kB] ;
            int64_t pB_end   = (kB < 0) ? -1 : Bp [kB+1] ;

            double ckwork = Cwork [k+1] - Cwork [k] ;
            // printf ("ckwork %g target_task_size %g\n",
                // ckwork, target_task_size) ;
            int nfine = ckwork / target_task_size ;
            nfine = GB_IMAX (nfine, 1) ;
            // printf ("nfine %d\n", nfine) ;

            //------------------------------------------------------------------
            // create the fine-grain tasks
            //------------------------------------------------------------------

            if (nfine == 1)
            {

                //--------------------------------------------------------------
                // this is a single coarse task for all of vector k
                //--------------------------------------------------------------

                TaskList [ntasks].kfirst = k ;
                TaskList [ntasks].klast  = k ;
                ntasks++ ;

            }
            else
            {

                //--------------------------------------------------------------
                // slice vector k into nfine fine tasks
                //--------------------------------------------------------------

                // first fine task starts at the top of vector k
                ASSERT (ntasks < max_ntasks) ;
                TaskList [ntasks].kfirst = k ;
                TaskList [ntasks].klast  = -1 ; // this is a fine task
                TaskList [ntasks].pA = pA_start ;
                TaskList [ntasks].pB = pB_start ;
                ntasks++ ;

                for (int tfine = 1 ; tfine < nfine ; tfine++)
                {
                    double target_work = ((nfine-tfine) * ckwork) / nfine ;
// printf ("tfine %d target %g\n", tfine, target_work) ;
                    int64_t i, pA, pB ;
                    GB_slice_vector (&i, &pA, &pB,
                        pA_start, pA_end, Ai, pB_start, pB_end, Bi,
                        vlen, target_work) ;
// printf ("    i "GBd" pA "GBd" pA "GBd"\n", i, pA, pB) ;

                    // tfine task starts at pA and pB 
                    ASSERT (ntasks < max_ntasks) ;
                    TaskList [ntasks].kfirst = k ;
                    TaskList [ntasks].klast  = -1 ; // this is a fine task
                    TaskList [ntasks].pA = pA ;
                    TaskList [ntasks].pB = pB ;
                    ntasks++ ;
                }

                // Terminate the last fine task.  This space will also be used
                // by the next task in the TaskList.  If the next task is a
                // fine task, it will operate on vector k+1, and its pA_start
                // will equal the pA_end of vector A(:,k), and likewise for B.
                // In that case, TaskList [t+1].pA and pB are both the end of
                // the prior task t, and the start of task t+1.  If the next
                // task t+1 is a coarse task, it will ignore its TaskList
                // [t+1].pA and pB, so this space can be used to termainte the
                // TaskList [t].
                TaskList [ntasks].pA = pA_end ;
                TaskList [ntasks].pB = pB_end ;
            }
        }
    }

    #if 0
    printf ("\nnthreads %d ntasks %d\n", nthreads, ntasks) ;
    for (int t = 0 ; t < ntasks ; t++)
    {
        printf ("Task %d: kfirst "GBd" klast "GBd" pA "GBd" pB "GBd
            " pC "GBd"\n", t,
            TaskList [t].kfirst,
            TaskList [t].klast,
            TaskList [t].pA,
            TaskList [t].pB,
            TaskList [t].pC) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_MEMORY (Cwork, Cnvec+1, sizeof (int64_t)) ;
    (*p_TaskList  ) = TaskList ;
    (*p_max_ntasks) = max_ntasks ;
    (*p_ntasks    ) = ntasks ;
    return (GrB_SUCCESS) ;
}

