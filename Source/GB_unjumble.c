//------------------------------------------------------------------------------
// GB_unjumble: unjumble the vectors of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_sort.h"

GrB_Info GB_unjumble        // unjumble a matrix
(
    GrB_Matrix A,           // matrix to unjumble
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A to unjumble", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;      // zombies must be killed first
    ASSERT (GB_PENDING_OK (A)) ;    // pending tuples are not modified

    if (!A->jumbled)
    {
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int64_t anvec = A->nvec ;
    const int64_t anz = GB_NNZ (A) ;
    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const size_t asize = A->type->size ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz + anvec, chunk, nthreads_max) ;
    int ntasks = (nthreads == 1) ? 1 : (32 * nthreads) ;
    ntasks = GB_IMIN (ntasks, anvec) ;
    ntasks = GB_IMAX (ntasks, 1) ;

    //--------------------------------------------------------------------------
    // slice the work
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT A_slice = NULL ;   // size ntasks + 1
    if (!GB_pslice (&A_slice, Ap, anvec, ntasks))
    {
        // out of memory
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // sort the vectors
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task description
        //----------------------------------------------------------------------

        int64_t kfirst = A_slice [tid] ;
        int64_t klast  = A_slice [tid+1] ;

        //----------------------------------------------------------------------
        // sort vectors kfirst to klast
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k < klast ; k++)
        {

            //------------------------------------------------------------------
            // check if the vector needs sorting
            //------------------------------------------------------------------

            bool jumbled = false ;
            int64_t pA_start = Ap [k] ;     // ok: A is sparse
            int64_t pA_end   = Ap [k+1] ;
            int64_t ilast = -1 ;
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = Ai [pA] ;       // ok: A is sparse
                if (i < ilast)
                { 
                    jumbled = true ;
                    break ;
                }
                ilast = i ;
            }

            //------------------------------------------------------------------
            // sort the vector
            //------------------------------------------------------------------

            if (jumbled)
            { 
                GB_qsort_1b (A->i + pA_start, A->x + pA_start*asize, asize,
                    pA_end - pA_start) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE (A_slice) ;
    A->jumbled = false ;
    ASSERT_MATRIX_OK (A, "A unjumbled", GB0) ;
    return (GrB_SUCCESS) ;
}

