//------------------------------------------------------------------------------
// GB_map_pslice: find where each task starts its work in matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Each task t operates on C(:,kfirst:klast), where kfirst = kfirst_slice [t]
// to klast = klast_slice [t], inclusive.  The kfirst vector for task t may
// overlap with one or more tasks 0 to t-1, and the klast vector for task t may
// overlap with the vectors of tasks t+1 to ntasks-1.  If kfirst == klast, then
// this single vector may overlap any other task.  Task t contributes Wfirst
// [t] entries to C(:,kfirst), and (if kfirst < klast) Wlast [t] entries to
// C(:,klast).  These entries are always in task order.  That is, if tasks t
// and t+1 both contribute to the same vector C(:,k), then all entries of task
// come just before all entries of task t+1.

// This function computes C_pstart_slice [0..ntasks-1].  Task t starts at its
// vector C(:,kfirst), at the position pC = C_pstart_slice [t].  It always
// starts its last vector C(:,klast) at Cp [klast], so this does not need to be
// computed.

#include "GB.h"

void GB_map_pslice
(
    // output
    int64_t *C_pstart_slice,                // size ntasks
    // input
    const int64_t *restrict Cp,             // size cnvec+1
    const int64_t *restrict kfirst_slice,   // size ntasks
    const int64_t *restrict klast_slice,    // size ntasks
    const int64_t *restrict Wfirst,         // size ntasks
    const int64_t *restrict Wlast,          // size ntasks
    int ntasks                              // number of tasks
)
{

//  for (int taskid = 0 ; taskid < ntasks ; taskid++)
//  {
//      printf ("\ntaskid %d of %d\n", taskid, ntasks) ;
//      printf ("   kfirst %d\n", kfirst_slice [taskid]) ;
//      printf ("   klast  %d\n", klast_slice [taskid]) ;
//      printf ("   Wfirst %d\n", Wfirst [taskid]) ;
//      printf ("   Wlast  %d\n", Wlast  [taskid]) ;
//  }

    int64_t kprior = -1 ;
    int64_t pC = 0 ;

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {
//      printf ("\ntaskid is %d\n", taskid) ;

        int64_t k = kfirst_slice [taskid] ;

        if (kprior < k)
        {
            // Task taskid is the first one to do work on C(:,k), so it starts
            // at Cp [k], and it contributes Wfirst [taskid] entries to C(:,k)
//          printf ("   kprior "GBd"\n", kprior) ;
//          printf ("   k "GBd"\n", k) ;
            pC = Cp [k] ;
            kprior = k ;
//          printf ("   pC at first "GBd"\n", pC) ;
        }

        // Task taskid contributes Wfirst [taskid] entries to C(:,k)
        C_pstart_slice [taskid] = pC ;
//      printf ("   C_pstart "GBd"\n", pC) ;
        pC += Wfirst [taskid] ;
//      printf ("   pC now  "GBd"\n", pC) ;

        int64_t klast = klast_slice [taskid] ;
        if (k < klast)
        {
            // Task taskid is the last to contribute to C(:,k).
//          printf ("   k "GBd"\n", k) ;
//          printf ("   klast  "GBd"\n", klast ) ;
//          printf ("   pC "GBd" should be "GBd"\n", pC, Cp [k+1]) ;
            ASSERT (pC == Cp [k+1]) ;
            // Task taskid contributes the first Wlast [taskid] entries
            // to C(:,klast), so the next task taskid+1 starts at this
            // location, if its first vector is klast of this task.
            pC = Cp [klast] + Wlast [taskid] ;
            kprior = klast ;
        }
    }

//  for (int taskid = 0 ; taskid < ntasks ; taskid++)
//  {
//      printf ("\ntaskid %d of %d\n", taskid, ntasks) ;
//      printf ("   C_pstart_slice  %d\n", C_pstart_slice [taskid]) ;
//  }
}

