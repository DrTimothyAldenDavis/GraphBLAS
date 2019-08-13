//------------------------------------------------------------------------------
// GB_matlab_helper.c: helper functions for MATLAB interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// These functions are only used by the MATLAB interface for
// SuiteSparse:GraphBLAS.

#include "GB_matlab_helper.h"

//------------------------------------------------------------------------------
// GB_matlab_helper1: convert 0-based indices to 1-based
//------------------------------------------------------------------------------

void GB_matlab_helper1      // convert zero-based indices to one-based
(
    double *I_double,       // output array
    const GrB_Index *I,     // input array
    int64_t nvals           // size of input and output arrays
)
{

    // determine the number of threads to use
    int nthreads_max = GB_Global_nthreads_max_get ( ) ;
    double chunk = GB_Global_chunk_get ( ) ;
    int nthreads = GB_nthreads (nvals, chunk, nthreads_max) ;

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int64_t k = 0 ; k < nvals ; k++)
    {
        I_double [k] = (double) (I [k] + 1) ;
    }
}

//------------------------------------------------------------------------------
// GB_matlab_helper2: create structure for dense matrix
//------------------------------------------------------------------------------

void GB_matlab_helper2      // fill Xp and Xi for a dense matrix
(
    GrB_Index *Xp,          // size ncols+1
    GrB_Index *Xi,          // size nrows*ncols
    int64_t ncols,
    int64_t nrows
)
{

    // determine the number of threads to use
    int nthreads_max = GB_Global_nthreads_max_get ( ) ;
    double chunk = GB_Global_chunk_get ( ) ;
    int nthreads = GB_nthreads (ncols, chunk, nthreads_max) ;

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int64_t j = 0 ; j <= ncols ; j++)
    {
        Xp [j] = j * nrows ;
    }

    double work = ((double) ncols) * ((double) nrows) ;
    nthreads = GB_nthreads (work, chunk, nthreads_max) ;

    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        collapse(2)
    for (int64_t j = 0 ; j < ncols ; j++)
    {
        for (int64_t i = 0 ; i < nrows ; i++)
        {
            Xi [j * nrows + i] = i ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_matlab_helper3: convert 1-based indices to 0-based
//------------------------------------------------------------------------------

bool GB_matlab_helper3          // return true if OK, false on error
(
    int64_t *List,              // size len, output array
    double *List_double,        // size len, input array
    int64_t len,
    int64_t *List_max           // also compute the max entry in the list
)
{

    // determine the number of threads to use
    int nthreads_max = GB_Global_nthreads_max_get ( ) ;
    double chunk = GB_Global_chunk_get ( ) ;
    int nthreads = GB_nthreads (len, chunk, nthreads_max) ;

    bool ok = true ;
    int64_t listmax = -1 ;

    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(&&:ok) reduction(max:listmax)
    for (int64_t k = 0 ; k < len ; k++)
    {
        double x = List_double [k] ;
        int64_t i = (int64_t) x ;
        ok = ok && (x == (double) i) ;
        listmax = GB_IMAX (listmax, i) ;
        List [k] = i - 1 ;
    }

    (*List_max) = listmax ;
    return (ok) ;
}

//------------------------------------------------------------------------------
// GB_matlab_helper4: find the max entry in an index list
//------------------------------------------------------------------------------

int64_t GB_matlab_helper4       // find max (I) + 1
(
    const GrB_Index *I,         // array of size len
    const int64_t len
)
{

    // determine the number of threads to use
    int nthreads_max = GB_Global_nthreads_max_get ( ) ;
    double chunk = GB_Global_chunk_get ( ) ;
    int nthreads = GB_nthreads (len, chunk, nthreads_max) ;

    GrB_Index imax = 0 ;
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(max:imax)
    for (int64_t k = 0 ; k < len ; k++)
    {
        imax = GB_IMAX (imax, I [k]) ;
    }
    if (len > 0) imax++ ;
    return (imax) ;
}

