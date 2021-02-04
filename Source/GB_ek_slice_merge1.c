//------------------------------------------------------------------------------
// GB_ek_slice_merge1: merge column counts for a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input matrix A has been sliced via GB_ek_slice, and scanned to compute
// the counts of entries in each vector of C in Cp, Wfirst, and Wlast.  This
// phase finalizes the column counts, Cp, merging the results of each task.

// On input, Cp [k] has been partially computed.  Task tid operators on vector
// kfirst = kfirst_Aslice [tid] to klast = klast_Aslice [tid].  If kfirst < k <
// klast, then Cp [k] is the total count of entries in C(:,k).  Otherwise, the
// counts are held in Wfirst and Wlast, and Cp [k] is zero (or uninititalized).
// Wfirst [tid] is the number of entries in C(:,kfirst) constructed by task
// tid, and Wlast [tid] is the number of entries in C(:,klast) constructed by
// task tid.

// This function sums up the entries computed for C(:,k) by all tasks, so that
// on output, Cp [k] is the total count of entries in C(:,k).

#include "GB_ek_slice.h"

void GB_ek_slice_merge1     // merge column counts for the matrix C
(
    // input/output:
    int64_t *GB_RESTRICT Cp,                    // column counts
    // input:
    const int64_t *GB_RESTRICT Ap,              // A->p
    const int64_t avlen,                        // A->vlen
    const int64_t *GB_RESTRICT Wfirst,          // size ntasks
    const int64_t *GB_RESTRICT Wlast,           // size ntasks
    const int64_t *GB_RESTRICT pstart_Aslice,   // size ntasks
    const int64_t *GB_RESTRICT kfirst_Aslice,   // size ntasks
    const int64_t *GB_RESTRICT klast_Aslice,    // size ntasks
    const int ntasks                            // # of tasks
)
{

    int64_t kprior = -1 ;

    for (int tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // sum up the partial result that thread tid computed for kfirst
        //----------------------------------------------------------------------

        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;

        if (kfirst <= klast)
        {
            int64_t pA_start = pstart_Aslice [tid] ;
            int64_t pA_end   = GBP (Ap, kfirst+1, avlen) ;
            pA_end = GB_IMIN (pA_end, pstart_Aslice [tid+1]) ;
            if (pA_start < pA_end)
            {
                if (kprior < kfirst)
                { 
                    // This thread is the first one that did work on
                    // A(:,kfirst), so use it to start the reduction.
                    Cp [kfirst] = Wfirst [tid] ;
                }
                else
                { 
                    Cp [kfirst] += Wfirst [tid] ;
                }
                kprior = kfirst ;
            }
        }

        //----------------------------------------------------------------------
        // sum up the partial result that thread tid computed for klast
        //----------------------------------------------------------------------

        if (kfirst < klast)
        {
            int64_t pA_start = GBP (Ap, klast, avlen) ;
            int64_t pA_end   = pstart_Aslice [tid+1] ;
            if (pA_start < pA_end)
            {
                /* if */ ASSERT (kprior < klast) ;
                { 
                    // This thread is the first one that did work on
                    // A(:,klast), so use it to start the reduction.
                    Cp [klast] = Wlast [tid] ;
                }
                /*
                else
                {
                    // If kfirst < klast and A(:,klast is not empty, then this
                    // task is always the first one to do work on A(:,klast),
                    // so this case is never used.
                    ASSERT (GB_DEAD_CODE) ;
                    Cp [klast] += Wlast [tid] ;
                }
                */
                kprior = klast ;
            }
        }
    }
}

