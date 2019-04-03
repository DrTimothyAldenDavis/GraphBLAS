//------------------------------------------------------------------------------
// GB_reduce_to_scalar_template: s=reduce(A), reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Reduce a matrix to a scalar.

{

    const GB_ATYPE *restrict Ax = A->x ;
    const int64_t  *restrict Ai = A->i ;
    int64_t anz = GB_NNZ (A) ;

    if (nthreads == 1)
    {

        //----------------------------------------------------------------------
        // single thread
        //----------------------------------------------------------------------

        if (A->nzombies == 0)
        {

            //------------------------------------------------------------------
            // no zombiies
            //------------------------------------------------------------------

            for (int64_t p = 0 ; p < anz ; p++)
            {
                // s += A(i,j)
                ASSERT (GB_IS_NOT_ZOMBIE (Ai [p])) ;
                GB_REDUCE (s, Ax, p) ;
                // check for early exit
                GB_REDUCE_TERMINAL (s) ;
            }
        }
        else
        {

            //------------------------------------------------------------------
            // with zombies
            //------------------------------------------------------------------

            for (int64_t p = 0 ; p < anz ; p++)
            {
                // s += A(i,j) if the entry is not a zombie
                if (GB_IS_NOT_ZOMBIE (Ai [p]))
                {
                    GB_REDUCE (s, Ax, p) ;
                    // check for early exit
                    GB_REDUCE_TERMINAL (s) ;
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // multiple threads
        //----------------------------------------------------------------------

        GB_REDUCE_WORKSPACE (w, nthreads) ;

        if (A->nzombies == 0)
        {

            //------------------------------------------------------------------
            // no zombies
            //------------------------------------------------------------------

            // each thread reduces its own part in parallel
            #pragma omp parallel for num_threads(nthreads)
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                GB_REDUCE_INIT (t) ;
                int64_t pstart = (tid == 0) ? 0 :
                    (((tid  ) * (double) anz) / (double) nthreads) ;
                int64_t pend   = (tid == nthreads-1) ? anz :
                    (((tid+1) * (double) anz) / (double) nthreads) ;
                for (int64_t p = pstart ; p < pend ; p++)
                {
                    // s += A(i,j)
                    ASSERT (GB_IS_NOT_ZOMBIE (Ai [p])) ;
                    GB_REDUCE (t, Ax, p) ;
                    // check for early exit
                    GB_REDUCE_TERMINAL (t) ;
                }
                GB_REDUCE_WRAPUP (w, tid, t) ;
            }
            // sum up the results of each part using a single thread
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                GB_REDUCE_W (s, w, tid) ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // with zombies
            //------------------------------------------------------------------

            // each thread reduces its own part in parallel
            #pragma omp parallel for num_threads(nthreads)
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                GB_REDUCE_INIT (t) ;
                int64_t pstart = (tid == 0) ? 0 :
                    (((tid  ) * (double) anz) / (double) nthreads) ;
                int64_t pend   = (tid == nthreads-1) ? anz :
                    (((tid+1) * (double) anz) / (double) nthreads) ;
                for (int64_t p = pstart ; p < pend ; p++)
                {
                    // s += A(i,j)
                    if (GB_IS_NOT_ZOMBIE (Ai [p]))
                    {
                        GB_REDUCE (t, Ax, p) ;
                        // check for early exit
                        GB_REDUCE_TERMINAL (t) ;
                    }
                }
                GB_REDUCE_WRAPUP (w, tid, t) ;
            }
            // sum up the results of each part using a single thread
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                GB_REDUCE_W (s, w, tid) ;
            }
        }
    }
}

