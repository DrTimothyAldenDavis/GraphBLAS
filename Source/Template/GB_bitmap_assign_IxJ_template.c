//------------------------------------------------------------------------------
// GB_bitmap_assign_IxJ_template: iterate over all of C(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Iterate over all positions in the IxJ Cartesian product.  This is all
// entries C(i,j) where i is in the list I and j is in the list J.  This
// traversal occurs whether or not C(i,j) is an entry present in C.

{

    //--------------------------------------------------------------------------
    // create the tasks to iterate over IxJ
    //--------------------------------------------------------------------------

    int ntasks = 0, max_ntasks = 0, nthreads ;
    GB_task_struct *TaskList = NULL ;
    GB_OK (GB_subassign_IxJ_slice (&TaskList, &max_ntasks, &ntasks, &nthreads,
        I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon, Context)) ;

    //--------------------------------------------------------------------------
    // iterate over all IxJ
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        // see also GB_GET_IXJ_TASK_DESCRIPTOR ; this is unmodified
        int64_t kfirst = TaskList [taskid].kfirst ;
        int64_t klast  = TaskList [taskid].klast ;
        bool fine_task = (klast == -1) ;
        if (fine_task)
        {
            // a fine task operates on a slice of a single vector
            klast = kfirst ;
        }
        int64_t iA_start = 0, iA_end = nI ;
        if (fine_task)
        {
            iA_start = TaskList [taskid].pA ;
            iA_end   = TaskList [taskid].pA_end ;
        }

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t jA = kfirst ; jA <= klast ; jA++)
        {

            //------------------------------------------------------------------
            // get jC, the corresponding vector of C
            //------------------------------------------------------------------

            int64_t jC = GB_ijlist (J, jA, Jkind, Jcolon) ;

            //------------------------------------------------------------------
            // operate on C (I(iA_start,iA_end-1),jC)
            //------------------------------------------------------------------

            for (int64_t iA = iA_start ; iA < iA_end ; iA++)
            {
                int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;
                int64_t pC = iC + jC * cvlen ;
                GB_IXJ_WORK (pC) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workpace
    //--------------------------------------------------------------------------

    GB_FREE (TaskList) ;
}

#undef GB_GET_pM

