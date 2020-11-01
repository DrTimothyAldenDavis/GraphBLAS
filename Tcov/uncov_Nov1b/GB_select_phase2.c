//------------------------------------------------------------------------------
// GB_select_phase2: C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{
    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int64_t  *GB_RESTRICT Ap = A->p ;
    const int64_t  *GB_RESTRICT Ah = A->h ;
    const int64_t  *GB_RESTRICT Ai = A->i ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) A->x ;
    size_t asize = A->type->size ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;

    //--------------------------------------------------------------------------
    // C = select (A)
    //--------------------------------------------------------------------------

    int tid ;
// TODO    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        // if kfirst > klast then task tid does no work at all
        int64_t kfirst = kfirst_slice [tid] ;
        int64_t klast  = klast_slice  [tid] ;

        //----------------------------------------------------------------------
        // selection from vectors kfirst to klast
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) to be operated on by this task
            //------------------------------------------------------------------

            int64_t pA_start, pA_end, pC ;
            GB_get_pA_and_pC (&pA_start, &pA_end, &pC, tid, k, kfirst, klast,
                pstart_slice, C_pstart_slice, Cp, avlen, Ap, avlen) ;

            //------------------------------------------------------------------
            // compact Ai and Ax [pA_start ... pA_end-1] into Ci and Cx
            //------------------------------------------------------------------

            #if defined ( GB_ENTRY_SELECTOR )

                GB_GET_J ;
                for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                {
                    int64_t i = GBI (Ai, pA, avlen) ;
                    if (GB_TEST_VALUE_OF_ENTRY (pA))
                    {   GB_cov[1482]++ ;
// covered (1482): 19400354
                        ASSERT (pC >= Cp [k] && pC < Cp [k+1]) ; // ok: C sparse
                        Ci [pC] = i ;                            // ok: C sparse
                        // Cx [pC] = Ax [pA] ;
                        GB_SELECT_ENTRY (Cx, pC, Ax, pA) ;
                        pC++ ;
                    }
                }

            #elif defined ( GB_TRIU_SELECTOR ) \
              ||  defined ( GB_RESIZE_SELECTOR )

                // keep pA_start to Zp[k]-1
                int64_t p = GB_IMIN (Zp [k], pA_end) ;  // ok: Z is sparse
                int64_t mynz = p - pA_start ;
                if (mynz > 0)
                {   GB_cov[1483]++ ;
// covered (1483): 195205
                    ASSERT (pC >= Cp [k] && pC + mynz <= Cp [k+1]) ;
                    if (Ai != NULL)
                    {   GB_cov[1484]++ ;
// covered (1484): 195205
                        // A and C are both sparse or hypersparse
                        memcpy (Ci +pC, Ai +pA_start, mynz*sizeof (int64_t)) ;
                    }
                    else
                    {
                        // A is full and C is sparse
                        int64_t i_start = pA_start % avlen ;
                        for (int64_t s = 0 ; s < mynz ; s++)
                        {   GB_cov[1485]++ ;
// NOT COVERED (1485):
GB_GOTCHA ;
                            int64_t i = i_start + s ;
                            ASSERT (GBI (Ai, pA_start+s, avlen) == i) ;
                            Ci [pC+s] = i ;
                        }
                    }
                    memcpy (Cx +pC*asize, Ax +pA_start*asize, mynz*asize) ;
                }

            #elif defined ( GB_DIAG_SELECTOR )

                // task that owns the diagonal entry does this work
                int64_t p = Zp [k] ;
                if (pA_start <= p && p < pA_end)
                {   GB_cov[1486]++ ;
// covered (1486): 8106
                    ASSERT (pC >= Cp [k] && pC + 1 <= Cp [k+1]) ;
                    Ci [pC] = GBI (Ai, p, avlen) ;               // ok: C sparse
                    memcpy (Cx +pC*asize, Ax +p*asize, asize) ;
                }

            #elif defined ( GB_OFFDIAG_SELECTOR )

                // keep pA_start to Zp[k]-1
                int64_t p = GB_IMIN (Zp [k], pA_end) ;
                int64_t mynz = p - pA_start ;
                if (mynz > 0)
                {   GB_cov[1487]++ ;
// covered (1487): 84749
                    ASSERT (pC >= Cp [k] && pC + mynz <= Cp [k+1]) ;
                    if (Ai != NULL)
                    {   GB_cov[1488]++ ;
// covered (1488): 84749
                        // A and C are both sparse or hypersparse
                        memcpy (Ci +pC, Ai +pA_start, mynz*sizeof (int64_t)) ;
                    }
                    else
                    {
                        // A is full and C is sparse or hypersparse
                        int64_t i_start = pA_start % avlen ;
                        for (int64_t s = 0 ; s < mynz ; s++)
                        {   GB_cov[1489]++ ;
// NOT COVERED (1489):
GB_GOTCHA ;
                            int64_t i = i_start + s ;
                            ASSERT (GBI (Ai, pA_start+s, avlen) == i) ;
                            Ci [pC+s] = i ;
                        }
                    }

                    memcpy (Cx +pC*asize, Ax +pA_start*asize, mynz*asize) ;
                    pC += mynz ;
                }

                // keep Zp[k]+1 to pA_end-1
                p = GB_IMAX (Zp [k]+1, pA_start) ;
                mynz = pA_end - p ;
                if (mynz > 0)
                {   GB_cov[1490]++ ;
// covered (1490): 13727
                    ASSERT (pA_start <= p && p < pA_end) ;
                    ASSERT (pC >= Cp [k] && pC + mynz <= Cp [k+1]) ;
                    if (Ai != NULL)
                    {   GB_cov[1491]++ ;
// covered (1491): 13727
                        // A and C are both sparse or hypersparse
                        memcpy (Ci +pC, Ai +p, mynz*sizeof (int64_t)) ;
                    }
                    else
                    {
                        // A is full and C is sparse or hypersparse
                        int64_t i_start = p % avlen ;
                        for (int64_t s = 0 ; s < mynz ; s++)
                        {   GB_cov[1492]++ ;
// NOT COVERED (1492):
GB_GOTCHA ;
                            int64_t i = i_start + s ;
                            ASSERT (GBI (Ai, p+s, avlen) == i) ;
                            Ci [pC+s] = i ;
                        }
                    }

                    memcpy (Cx +pC*asize, Ax +p*asize, mynz*asize) ;
                }

            #elif defined ( GB_TRIL_SELECTOR )

                // keep Zp [k] to pA_end-1
                int64_t p = GB_IMAX (Zp [k], pA_start) ;
                int64_t mynz = pA_end - p ;
                if (mynz > 0)
                {   GB_cov[1493]++ ;
// covered (1493): 170367
                    ASSERT (pA_start <= p && p + mynz <= pA_end) ;
                    ASSERT (pC >= Cp [k] && pC + mynz <= Cp [k+1]) ;
                    if (Ai != NULL)
                    {   GB_cov[1494]++ ;
// covered (1494): 170367
                        // A and C are both sparse or hypersparse
                        memcpy (Ci +pC, Ai +p, mynz*sizeof (int64_t)) ;
                    }
                    else
                    {
                        // A is full and C is sparse or hypersparse
                        int64_t i_start = p % avlen ;
                        for (int64_t s = 0 ; s < mynz ; s++)
                        {   GB_cov[1495]++ ;
// NOT COVERED (1495):
                            int64_t i = i_start + s ;
                            ASSERT (GBI (Ai, p+s, avlen) == i) ;
                            Ci [pC+s] = i ;
                        }
                    }
                    memcpy (Cx +pC*asize, Ax +p*asize, mynz*asize) ;
                }

            #endif
        }
    }
}

