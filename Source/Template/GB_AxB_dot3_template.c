//------------------------------------------------------------------------------
// GB_AxB_dot3_template: C<M>=A'*B via dot products, where C is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C and M are both sparse or hyper, and C->h is a copy of M->h.
// M is present, and not complemented.  It may be valued or structural.

{

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:nzombies)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        int64_t kfirst = TaskList [tid].kfirst ;
        int64_t klast  = TaskList [tid].klast ;
        int64_t pC_first = TaskList [tid].pC ;
        int64_t pC_last  = TaskList [tid].pC_end ;
        int64_t task_nzombies = 0 ;
        int64_t bpleft = 0 ;    // ok: Ch is not jumbled

        //----------------------------------------------------------------------
        // compute all vectors in this task
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // get C(:,k) and M(:k)
            //------------------------------------------------------------------

            int64_t j = GBH (Ch, k) ;
            int64_t pC_start = Cp [k] ;
            int64_t pC_end   = Cp [k+1] ;
            if (k == kfirst)
            { 
                // First vector for task; may only be partially owned.
                pC_start = pC_first ;
                pC_end   = GB_IMIN (pC_end, pC_last) ;
            }
            else if (k == klast)
            { 
                // Last vector for task; may only be partially owned.
                pC_end   = pC_last ;
            }
            else
            { 
                // task completely owns this vector C(:,k).
            }

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            #if GB_B_IS_HYPER
                // B is hyper
                int64_t pB_start, pB_end ;
                GB_lookup (true, Bh, Bp, vlen, &bpleft, bnvec-1, j,
                    &pB_start, &pB_end) ;
            #elif GB_B_IS_SPARSE
                // B is sparse
                const int64_t pB_start = Bp [j] ;
                const int64_t pB_end = Bp [j+1] ;
            #else
                // B is bitmap or full
                const int64_t pB_start = j * vlen ;
            #endif

            #if (GB_B_IS_SPARSE || GB_B_IS_HYPER)
                const int64_t bjnz = pB_end - pB_start ;
                if (bjnz == 0)
                {
                    // no work to do if B(:,j) is empty, except for zombies
                    task_nzombies += (pC_end - pC_start) ;
                    for (int64_t pC = pC_start ; pC < pC_end ; pC++)
                    { 
                        // C(i,j) is a zombie
                        int64_t i = Mi [pC] ;
                        Ci [pC] = GB_FLIP (i) ;
                    }
                    continue ;
                }
                #if (GB_A_IS_SPARSE || GB_A_IS_HYPER)
                    // Both A and B are sparse; get first and last in B(:,j)
                    const int64_t ib_first = Bi [pB_start] ;
                    const int64_t ib_last  = Bi [pB_end-1] ;
                #endif
            #endif

            //------------------------------------------------------------------
            // C(:,j)<M(:,j)> = A(:,i)'*B(:,j)
            //------------------------------------------------------------------

            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            {

                //----------------------------------------------------------
                // get C(i,j) and M(i,j)
                //----------------------------------------------------------

                bool cij_exists = false ;

                // get the value of M(i,j)
                int64_t i = Mi [pC] ;
                if (GB_mcast (Mx, pC, msize))
                { 

                    //------------------------------------------------------
                    // the mask allows C(i,j) to be computed
                    //------------------------------------------------------

                    #if GB_A_IS_HYPER
                    // A is hyper
                    int64_t pA, pA_end ;
                    int64_t apleft = 0 ;    // M might be jumbled
                    GB_lookup (true, Ah, Ap, vlen, &apleft, anvec-1, i,
                        &pA, &pA_end) ;
                    const int64_t ainz = pA_end - pA ;
                    if (ainz > 0)
                    #elif GB_A_IS_SPARSE
                    // A is sparse
                    int64_t pA = Ap [i] ;
                    const int64_t pA_end = Ap [i+1] ;
                    const int64_t ainz = pA_end - pA ;
                    if (ainz > 0)
                    #else
                    // A is bitmap or full
                    const int64_t pA = i * vlen ;
                    #endif
                    { 
                        // C(i,j) = A(:,i)'*B(:,j)
                        #include "GB_AxB_dot_cij.c"
                    }
                }

                if (!GB_CIJ_EXISTS)
                { 
                    // C(i,j) is a zombie
                    task_nzombies++ ;
                    Ci [pC] = GB_FLIP (i) ;
                }
            }
        }

        //----------------------------------------------------------------------
        // sum up the zombies found by this task
        //----------------------------------------------------------------------

        nzombies += task_nzombies ; // TODO: just use nzombies
    }
}

#undef GB_A_IS_SPARSE
#undef GB_A_IS_HYPER
#undef GB_A_IS_BITMAP
#undef GB_A_IS_FULL
#undef GB_B_IS_SPARSE
#undef GB_B_IS_HYPER
#undef GB_B_IS_BITMAP
#undef GB_B_IS_FULL

