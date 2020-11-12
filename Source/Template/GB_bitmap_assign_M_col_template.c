//------------------------------------------------------------------------------
// GB_bitmap_assign_M_col_template:  traverse M for GB_COL_ASSIGN
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// M is a (C->vlen)-by-1 hypersparse or sparse matrix, for
// GrB_Row_assign (if C is CSR) or GrB_Col_assign (if C is CSC).

{
    int64_t jC = J [0] ;
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

            ASSERT (k == 0) ;
            ASSERT (GBH (Mh, k) == 0) ;
            int64_t pM_start, pM_end ;
            GB_get_pA (&pM_start, &pM_end, tid, k, kfirst,
                klast, pstart_Mslice, Mp, mvlen) ;

            //------------------------------------------------------------------
            // traverse over M(:,0), the kth vector of M
            //------------------------------------------------------------------

            // for col_assign: M is a single vector, jC = J [0]
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

