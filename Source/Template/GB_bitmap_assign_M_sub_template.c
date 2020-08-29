//------------------------------------------------------------------------------
// GB_bitmap_assign_M_sub_template:  traverse M for GB_SUBASSIGN
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// M is sparse or hypersparse, not bitmap or full.  C(I,J)<M>= ... is being
// computed (or !M), and all entries in M are traversed.  For a given entry
// M(iM,jM) in the mask, at location pM.  The entry C(iC,jC) is at location pC,
// where iC = I [iM] and jC = J [jM].  The matrix C is bitmap or full.

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

            // for subassign, M has same size as C(I,J) and A.
            int64_t jC = GB_ijlist (J, jM, Jkind, Jcolon) ;
            int64_t pC0 = jC * cvlen ;

            for (int64_t pM = pM_start ; pM < pM_end ; pM++)
            {
                bool mij = GB_mcast (Mx, pM, msize) ;
                if (mij)
                { 
                    int64_t iM = Mi [pM] ;          // ok: M is sparse
                    int64_t iC = GB_ijlist (I, iM, Ikind, Icolon) ;
                    int64_t pC = iC + pC0 ;
                    GB_MASK_WORK (pC) ;             // operate on Cx [pC]
                }
            }
        }
    }
}

