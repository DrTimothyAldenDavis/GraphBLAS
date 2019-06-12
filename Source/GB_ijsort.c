//------------------------------------------------------------------------------
// GB_ijsort:  sort an index array I and remove duplicates
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// PARALLEL: TODO

#include "GB.h"

GrB_Info GB_ijsort
(
    const GrB_Index *I, // index array of size ni
    int64_t *p_ni,      // input: size of I, output: number of indices in I2
    GrB_Index **p_I2,   // output array of size ni, where I2 [0..ni2-1]
                        // contains the sorted indices with duplicates removed.
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (I != NULL) ;
    ASSERT (p_ni != NULL) ;
    ASSERT (p_I2 != NULL) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Index *I2 = NULL ;
    int64_t ni = *p_ni ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (ni, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // allocate the new list
    //--------------------------------------------------------------------------

    GB_MALLOC_MEMORY (I2, ni, sizeof (GrB_Index)) ;
    if (I2 == NULL)
    { 
        // out of memory
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // copy I into I2 and sort it
    //--------------------------------------------------------------------------

    GB_memcpy (I2, I, ni * sizeof (GrB_Index), nthreads) ;
    GB_qsort_1a ((int64_t *) I2, ni, Context) ;

    //--------------------------------------------------------------------------
    // remove duplicates from I2
    //--------------------------------------------------------------------------

    int64_t ni2 = 1 ;
    for (int64_t k = 1 ; k < ni ; k++)
    {
        if (I2 [ni2-1] != I2 [k])
        { 
            I2 [ni2++] = I2 [k] ;
        }
    }

    //--------------------------------------------------------------------------
    // return the new sorted list
    //--------------------------------------------------------------------------

    *p_I2 = I2 ;        // I2 has size ni, but only I2 [0..ni2-1] is defined
    *p_ni = ni2 ;

    return (GrB_SUCCESS) ;
}

