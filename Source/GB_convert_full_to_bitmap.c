//------------------------------------------------------------------------------
// GB_convert_full_to_bitmap: convert a matrix from full to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_convert_full_to_bitmap      // convert matrix from full to bitmap
(
    GrB_Matrix A,               // matrix to convert from full to bitmap
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting full to bitmap", GB0) ;
    ASSERT (GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    GBURBLE ("(full to bitmap) ") ;

    //--------------------------------------------------------------------------
    // allocate A->b
    //--------------------------------------------------------------------------

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;
    int64_t anz = avdim * avlen ;
    ASSERT (GB_Index_multiply (&anz, avdim, avlen) == true) ;

    A->b = GB_MALLOC (anz, int8_t) ;
    if (A->b == NULL)
    { 
        // out of memory
        GB_phbix_free (A) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // fill the A->b bitmap
    //--------------------------------------------------------------------------

    int8_t *GB_RESTRICT Ab = A->b ;

    int64_t taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (taskid = 0 ; taskid < nthreads ; taskid++)
    {
        // Ab [p1:p2-1] = 1
        int64_t pstart, pend ;
        GB_PARTITION (pstart, pend, anz, taskid, nthreads) ;
        memset (Ab + pstart, 1, pend - pstart + 1) ;
    }

    A->nvals = anz ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from full to bitmap", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

