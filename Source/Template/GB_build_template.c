//------------------------------------------------------------------------------
// GB_build_template: T=build(S), and assemble any duplicate tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------


{
    if (ndupl == 0)
    {

        //----------------------------------------------------------------------
        // no duplicates, just permute S into Tx
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            int64_t tstart = tstart_slice [tid] ;
            int64_t tend   = tstart_slice [tid+1] ;
            for (int64_t t = tstart ; t < tend ; t++)
            {
                // Tx [t] = S [k] ;
                int64_t k = (kwork == NULL) ? t : kwork [t] ;
                GB_BUILD_COPY (Tx, t, S, k) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // assemble duplicates
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            int64_t my_tnz = tnz_slice [tid] ;
            int64_t tstart = tstart_slice [tid] ;
            int64_t tend   = tstart_slice [tid+1] ;

            // find the first unique tuple owned by this slice
            int64_t t ;
            for (t = tstart ; t < tend ; t++)
            {
                // get the tuple and break if it is not a duplicate
                if (iwork [t] >= 0) break ;
            }

            // scan all tuples and assemble any duplicates
            for ( ; t < tend ; t++)
            {
                // get the t-th tuple, a unique tuple
                int64_t i = iwork [t] ;
                int64_t k = (kwork == NULL) ? t : kwork [t] ;
                ASSERT (i >= 0) ;
                // Tx [my_tnz] = S [k] ;
                GB_BUILD_COPY (Tx, my_tnz, S, k) ;
                Ti [my_tnz] = i ;

                // assemble all duplicates that follow it.  This may assemble
                // the first duplicates in the next slice (up to but not
                // including the first unique tuple in the subsequent slice).
                for ( ; t+1 < nvals && iwork [t+1] < 0 ; t++)
                {
                    // assemble the duplicate tuple
                    int64_t k = (kwork == NULL) ? (t+1) : kwork [t+1] ;
                    // duplicate entry: Tx [my_tnz] += S [k]
                    GB_BUILD_OP (Tx, my_tnz, S, k) ;
                }
                my_tnz++ ;
            }
        }
    }
}

