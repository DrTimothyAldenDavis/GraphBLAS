//------------------------------------------------------------------------------
// GB_pslice: partition A->p by # of entries, for a parallel loop
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A->p [0..A->nvec] is a monotonically increasing cumulative sum of the
// entries in each vector of A.  This function slices Ap so that each chunk has
// the same number of entries.  Returns the # of threads that can be
// effectively used.

#include "GB.h"

void GB_pslice                  // find how to slice A->p by # of entries
(
    int64_t *Slice,             // size nthreads+1
    const GrB_Matrix A,
    const int nthreads          // # of threads
)
{

    const int64_t *restrict Ap = A->p ;
    int64_t anz = GB_NNZ (A) ;
    int64_t anvec = A->nvec ;
    Slice [0] = 0 ;
    if (Ap == NULL || anvec == 0 || nthreads <= 1 || anz == 0)
    {
        // matrix is empty, or a single thread is used
        for (int tid = 1 ; tid < nthreads ; tid++)
        {
            Slice [tid] = 0 ;
        }
    }
    else
    {
        // slice Ap by # of entries
        int64_t pleft = 0 ;
        for (int tid = 1 ; tid < nthreads ; tid++)
        { 
            // binary search to find k so that Ap [k] == (tid * anz) /
            // nthreads.  The exact value will not typically not be found;
            // just pick what the binary search comes up with.
            int64_t nz = ((tid * (double) anz) / (double) nthreads) ;
            int64_t pright = anvec ;
            GB_BINARY_TRIM_SEARCH (nz, Ap, pleft, pright) ;
            Slice [tid] = pleft ;
        }
    }
    Slice [nthreads] = anvec ;
}

