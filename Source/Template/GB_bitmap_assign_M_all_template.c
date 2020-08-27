{
    int tid ;
    #pragma omp parallel for num_threads(mthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < mtasks ; tid++)
    {
        int64_t kfirst = kfirst_Mslice [tid] ;
        int64_t klast  = klast_Mslice  [tid] ;

        //----------------------------------------------------------------------
        // traverse over M (:,kfirst:klast)
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of M(:,k) for this task
            //------------------------------------------------------------------

            int64_t jM = GBH (Mh, k) ;
            int64_t pM_start, pM_end ;
            GB_get_pA_and_pC (&pM_start, &pM_end, NULL, tid, k, kfirst,
                klast, pstart_Mslice, NULL, NULL, 0, Mp, mvlen) ;

            //------------------------------------------------------------------
            // traverse over M(:,jM), the kth vector of M
            //------------------------------------------------------------------

            // for assign: M is a matrix the same size as C
            int64_t jC = jM ;
            for (int64_t pM = pM_start ; pM < pM_end ; pM++)
            {
                bool mij = GB_mcast (Mx, pM, msize) ;
                if (mij)
                { 
                    int64_t iC = Mi [pM] ;          // ok: M is sparse
                    int64_t pC = iC + jC * cvlen ;
                    GB_MASK_WORK (pC) ;
                }
            }
        }
    }
}

