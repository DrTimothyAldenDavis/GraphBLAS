//------------------------------------------------------------------------------
// GB_AxB_saxpy4_tasks: construct tasks for saxpy4 and bitmap_saxpy
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_AxB_saxpy4_tasks constructs the tasks for GB_AxB_saxpy4, and for
// GB_AxB_bitmap_saxpy when A is sparse/hyper and B is bitmap/full.

#include "GB_AxB_saxpy.h"

GrB_Info GB_AxB_saxpy4_tasks
(
    // output
    int *p_ntasks,                  // # of tasks to use
    int *p_nthreads,                // # of threads to use
    int *p_nfine_tasks_per_vector,  // # of tasks per vector (fine case only)
    bool *p_use_coarse_tasks,       // if true, use coarse tasks
    bool *p_use_atomics,            // if true, use atomics
    // input
    int64_t anz,                    // # of entries in A (sparse or hyper)
    int64_t bnz,                    // # of entries held in B
    int64_t bvdim,                  // # of vectors of B (bitmap or full)
    int64_t cvlen,                  // # of vectors of C (bitmap or full)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // determine the work to do
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    double work = ((double) anz) * (double) bvdim ;
    int nthreads = GB_nthreads (work, chunk, nthreads_max) ;
    int nfine_tasks_per_vector = 0, ntasks ;
    bool use_coarse_tasks, use_atomics = false ;

    //--------------------------------------------------------------------------
    // create the tasks
    //--------------------------------------------------------------------------

    if (nthreads == 1 || bvdim == 0)
    { 

        //----------------------------------------------------------------------
        // do the entire computation with a single thread, with coarse task
        //----------------------------------------------------------------------

        ntasks = 1 ;
        use_coarse_tasks = true ;
        GBURBLE ("(coarse, threads: 1) ") ;

    }
    else if (nthreads <= bvdim)
    { 

        //----------------------------------------------------------------------
        // all tasks are coarse
        //----------------------------------------------------------------------

        // Each coarse task does 1 or more whole vectors of B
        ntasks = GB_IMIN (bvdim, 2 * nthreads) ;
        nthreads = GB_IMIN (ntasks, nthreads) ;
        use_coarse_tasks = true ;
        GBURBLE ("(coarse, threads: %d, tasks %d) ", nthreads, ntasks) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // use fine tasks, unless there are not enough threads for atomic method
        //----------------------------------------------------------------------

        // Each task does a slice of a single vector of B, and each vector of B
        // is handled by the same # of fine tasks.  Determine if atomics are
        // to be used or not.

        double cnz = ((double) cvlen) * ((double) bvdim) ;
        double intensity = work / fmax (cnz, 1) ;
        double workspace = ((double) cvlen) * ((double) nthreads) ;
        double relwspace = workspace / fmax (anz + bnz + cnz, 1) ;
        GBURBLE ("(threads: %d, relwspace: %0.3g, intensity: %0.3g",
            nthreads, relwspace, intensity) ;
        if ((intensity > 64 && relwspace < 0.05) ||
            (intensity > 16 && intensity <= 64 && relwspace < 0.50))
        { 
            // fine non-atomic method with workspace
            use_coarse_tasks = false ;
            ntasks = nthreads ;
            GBURBLE (": fine non-atomic, ") ;
        }
        else if (nthreads <= 2*bvdim)
        { 
            // not enough threads for good performance in the fine atomic
            // method; disable atomics and use the coarse method instead, with
            // one task per vector of B
            use_coarse_tasks = true ;
            ntasks = bvdim ;
            nthreads = GB_IMIN (ntasks, nthreads) ;
            GBURBLE (": coarse, threads used: %d, tasks: %d (one per vector)) ",
                nthreads, ntasks) ;
        }
        else
        { 
            // fine atomic method, with no workspace
            use_coarse_tasks = false ;
            use_atomics = true ;
            ntasks = 2 * nthreads ;         // FIXME: try more tasks
            GBURBLE (": fine atomic, ") ;
        }

        if (!use_coarse_tasks)
        { 
            nfine_tasks_per_vector = ceil ((double) ntasks / (double) bvdim) ;
            ntasks = bvdim * nfine_tasks_per_vector ;
            ASSERT (nfine_tasks_per_vector > 1) ;
            GBURBLE ("tasks: %d, tasks per vector: %d) ", ntasks,
                nfine_tasks_per_vector) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_ntasks) = ntasks ;
    (*p_nthreads) = nthreads ;
    (*p_nfine_tasks_per_vector) = nfine_tasks_per_vector ;
    (*p_use_coarse_tasks) = use_coarse_tasks ;
    (*p_use_atomics) = use_atomics ;
}

