//------------------------------------------------------------------------------
// GB_reduce_to_scalar_template: s=reduce(A), reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Reduce a matrix to a scalar

// PARALLEL: done

// TODO add simd vectorization for non-terminal monoids.  Particular max, min

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
            // no zombies
            //------------------------------------------------------------------

            for (int64_t p = 0 ; p < anz ; p++)
            {
                ASSERT (GB_IS_NOT_ZOMBIE (Ai [p])) ;
                // s += (ztype) Ax [p]
                GB_ADD_CAST_ARRAY_TO_SCALAR (s, Ax, p) ;
                // check for early exit
                GB_BREAK_IF_TERMINAL (s) ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // with zombies
            //------------------------------------------------------------------

            for (int64_t p = 0 ; p < anz ; p++)
            {
                if (GB_IS_NOT_ZOMBIE (Ai [p]))
                {
                    // s += (ztype) Ax [p]
                    GB_ADD_CAST_ARRAY_TO_SCALAR (s, Ax, p) ;
                    // check for early exit
                    GB_BREAK_IF_TERMINAL (s) ;
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // create workspace for multiple threads
        //----------------------------------------------------------------------

        // ztype W [nthreads] ;
        GB_REDUCTION_WORKSPACE (W, nthreads) ;
        ASSERT (nthreads <= anz) ;

        //----------------------------------------------------------------------
        // each thread reduces its own slice in parallel
        //----------------------------------------------------------------------

        if (A->nzombies == 0)
        {

            //------------------------------------------------------------------
            // no zombies
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                int64_t pstart, pend ;
                GB_PARTITION (pstart, pend, anz, tid, nthreads) ;
                // no slice is empty
                ASSERT (pstart < pend) ;

                // ztype t = identity
                GB_SCALAR_IDENTITY (t) ;

                for (int64_t p = pstart ; p < pend ; p++)
                {
                    ASSERT (GB_IS_NOT_ZOMBIE (Ai [p])) ;
                    // t += (ztype) Ax [p], with typecast
                    GB_ADD_CAST_ARRAY_TO_SCALAR (t, Ax, p) ;
                    // check for early exit
                    GB_BREAK_IF_TERMINAL (t) ;
                }
                // W [tid] = t, no typecast
                GB_COPY_SCALAR_TO_ARRAY (W, tid, t) ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // with zombies
            //------------------------------------------------------------------

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                int64_t pstart, pend ;
                GB_PARTITION (pstart, pend, anz, tid, nthreads) ;
                // no slice is empty
                ASSERT (pstart < pend) ;

                // ztype t = identity
                GB_SCALAR_IDENTITY (t) ;

                for (int64_t p = pstart ; p < pend ; p++)
                {
                    if (GB_IS_NOT_ZOMBIE (Ai [p]))
                    {
                        // t += (ztype) Ax [p], with typecast
                        GB_ADD_CAST_ARRAY_TO_SCALAR (t, Ax, p) ;
                        // check for early exit
                        GB_BREAK_IF_TERMINAL (t) ;
                    }
                }
                // W [tid] = t, no typecast
                GB_COPY_SCALAR_TO_ARRAY (W, tid, t) ;
            }
        }

        //----------------------------------------------------------------------
        // sum up the results of each slice using a single thread
        //----------------------------------------------------------------------

        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            // s += W [tid], no typecast
            GB_ADD_ARRAY_TO_SCALAR (s, W, tid) ;
        }
    }
}

