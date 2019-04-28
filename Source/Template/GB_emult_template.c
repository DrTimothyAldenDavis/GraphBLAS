//------------------------------------------------------------------------------
// GB_emult_template:  phase1 and phase2 for C=A.*B, C<M>=A.*B, and C<!M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, M, and C
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    int64_t vlen = A->vlen ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;

    const int64_t *restrict Mp = NULL ;
    const int64_t *restrict Mh = NULL ;
    const int64_t *restrict Mi = NULL ;
    const GB_void *restrict Mx = NULL ;
    GB_cast_function cast_M = NULL ;
    size_t msize = 0 ;
    int64_t Mnvec = 0 ;
    bool M_is_hyper = false ;
    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mi = M->i ;
        Mx = M->x ;
        cast_M = GB_cast_factory (GB_BOOL_code, M->type->code) ;
        msize = M->type->size ;
        Mnvec = M->nvec ;
        M_is_hyper = M->is_hyper ;
    }

    #if !defined ( GB_PHASE_1_OF_2 )
    const GB_ATYPE *restrict Ax = A->x ;
    const GB_ATYPE *restrict Bx = B->x ;
    const int64_t  *restrict Cp = C->p ;
    const int64_t  *restrict Ch = C->h ;
          int64_t  *restrict Ci = C->i ;
          GB_CTYPE *restrict Cx = C->x ;
    int64_t Cnvec = C->nvec ;
    #endif

    //--------------------------------------------------------------------------
    // determine the # of teams and their sizes
    //--------------------------------------------------------------------------

    int nteams ;                // # of thread teams to use across the vectors
    int nthreads_per_team ;     // size of thread team to use inside each vector
    GB_teams (Cnvec, nthreads, &nteams, &nthreads_per_team) ;

    //--------------------------------------------------------------------------
    // phase1: count nnz in each C(:,j);  phase2: compute C
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nteams) schedule(guided,1)
    for (int64_t k = 0 ; k < Cnvec ; k++)
    {

        //----------------------------------------------------------------------
        // get j, the kth vector of C
        //----------------------------------------------------------------------

        int64_t j = (Ch == NULL) ? k : Ch [k] ;

        #if defined ( GB_PHASE_1_OF_2 )
        int64_t cjnz = 0 ;
        #else
        int64_t pC     = Cp [k] ;
        int64_t pC_end = Cp [k+1] ;
        int64_t cjnz = pC_end - pC ;
        if (cjnz == 0) continue ;
        #endif

        //----------------------------------------------------------------------
        // get A(:,j)
        //----------------------------------------------------------------------

        int64_t pA = -1 ;
        int64_t pA_end = -1 ;
        int64_t kA = (Ch == Ah) ? k : ((C_to_A == NULL) ? j : C_to_A [k]) ;
        ASSERT (kA >= -1 && kA < A->nvec) ;
        if (kA >= 0)
        { 
            pA     = Ap [kA] ;
            pA_end = Ap [kA+1] ;
        }
        int64_t ajnz = pA_end - pA ;    // nnz (A (:,j))

        //----------------------------------------------------------------------
        // get B(:,j)
        //----------------------------------------------------------------------

        int64_t pB = -1 ;
        int64_t pB_end = -1 ;
        int64_t kB = (Ch == Bh) ? k : ((C_to_B == NULL) ? j : C_to_B [k]) ;
        ASSERT (kB >= -1 && kB < B->nvec) ;
        if (kB >= 0)
        { 
            pB     = Bp [kB] ;
            pB_end = Bp [kB+1] ;
        }
        int64_t bjnz = pB_end - pB ;    // nnz (B (:,j))

        //----------------------------------------------------------------------
        // get M(:,j)
        //----------------------------------------------------------------------

        int64_t pM = -1 ;
        int64_t pM_end = -1 ;
        if (M != NULL)
        {
            int64_t kM = (Ch == Mh) ? k : ((C_to_M == NULL) ? j : C_to_M [k]) ;
            ASSERT (kM >= -1 && kM < M->nvec) ;
            if (kM >= 0)
            { 
                pM     = Mp [kM] ;
                pM_end = Mp [kM+1] ;
            }
        }
        int64_t mjnz = pM_end - pM ;    // nnz (M (:,j))

        //----------------------------------------------------------------------
        // phase1: count nnz (C (:,j)), phase2: compute C(:,j)
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_1_OF_2 )

        if (M != NULL && mjnz == 0 && !Mask_comp)
        { 

            //------------------------------------------------------------------
            // M(:,j) is empty and not complemented
            //------------------------------------------------------------------

            // C(:,j) is empty, regardless of A(:,j) and B(:,j)
            ;

        }
        else if (ajnz == 0 || bjnz == 0)
        { 

            //------------------------------------------------------------------
            // A(:,j) and/or B(:,j) are empty 
            //------------------------------------------------------------------

            ;

        }
        else if (Ai [pA_end-1] < Bi [pB] || Bi [pB_end-1] < Ai [pA])
        { 

            //------------------------------------------------------------------
            // intersection of A(:,j) and B(:,j) is empty
            //------------------------------------------------------------------

            // the last entry of A(:,j) comes before the first entry
            // of B(:,j), or visa versa
            ;

        }
        else

        #endif

        if (M == NULL || (M != NULL && mjnz == 0 && Mask_comp))
        {

            //------------------------------------------------------------------
            // No mask, or M(:,j) is empty and complemented
            //------------------------------------------------------------------

            // if present, M(:,j) is ignored since !M(:,j) is all true

            if (ajnz == vlen && bjnz == vlen)
            {

                //--------------------------------------------------------------
                // A(:,j) and B(:,j) dense
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = vlen ;
                #else
                ASSERT (cjnz == vlen) ;
                int nth = GB_nthreads (vlen, 4096, nthreads_per_team) ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (int64_t i = 0 ; i < vlen ; i++)
                { 
                    Ci [pC + i] = i ;
                    GB_GETA (aij, Ax, pA + i) ;
                    GB_GETB (bij, Bx, pB + i) ;
                    GB_BINOP (GB_CX (pC + i), aij, bij) ;
                }
                #endif

            }
            else if (ajnz == vlen)
            {

                //--------------------------------------------------------------
                // A(:,j) is dense, B(:,j) is sparse
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = bjnz ;
                #else
                ASSERT (cjnz == bjnz) ;
                int nth = GB_nthreads (bjnz, 1024, nthreads_per_team) ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (int64_t p = 0 ; p < bjnz ; p++)
                { 
                    int64_t i = Bi [pB + p] ;
                    Ci [pC + p] = i ;
                    GB_GETA (aij, Ax, pA + i) ;
                    GB_GETB (bij, Bx, pB + p) ;
                    GB_BINOP (GB_CX (pC + p), aij, bij) ;
                }
                #endif

            }
            else if (bjnz == vlen)
            {

                //--------------------------------------------------------------
                // A(:,j) is sparse, B(:,j) is dense
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = ajnz ;
                #else
                ASSERT (cjnz == ajnz) ;
                int nth = GB_nthreads (ajnz, 1024, nthreads_per_team) ;
                #pragma omp parallel for num_threads(nth) schedule(static)
                for (int64_t p = 0 ; p < ajnz ; p++)
                { 
                    int64_t i = Ai [pA + p] ;
                    Ci [pC + p] = i ;
                    GB_GETA (aij, Ax, pA + p) ;
                    GB_GETB (bij, Bx, pB + i) ;
                    GB_BINOP (GB_CX (pC + p), aij, bij) ;
                }
                #endif

            }
            else if (ajnz > 32 * bjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than B(:,j)
                //--------------------------------------------------------------

                // TODO: phase1: use a reduction on cjnz

                for ( ; pB < pB_end ; pB++)
                {
                    int64_t i = Bi [pB] ;
                    // find i in A(:,j)
                    int64_t pright = pA_end - 1 ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Ai, pA, pright, found) ;
                    if (found)
                    { 
                        #if defined ( GB_PHASE_1_OF_2 )
                        cjnz++ ;
                        #else
                        ASSERT (pC < pC_end) ;
                        Ci [pC] = i ;
                        GB_GETA (aij, Ax, pA) ;
                        GB_GETB (bij, Bx, pB) ;
                        GB_BINOP (GB_CX (pC), aij, bij) ;
                        pC++ ;
                        #endif
                    }
                }
                #if !defined ( GB_PHASE_1_OF_2 )
                ASSERT (pC == pC_end) ;
                #endif

            }
            else if (bjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // B(:,j) is much denser than A(:,j)
                //--------------------------------------------------------------

                // TODO: phase1: use a reduction on cjnz

                for ( ; pA < pA_end ; pA++)
                {
                    int64_t i = Ai [pA] ;
                    // find i in B(:,j)
                    int64_t pright = pB_end - 1 ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Bi, pB, pright, found) ;
                    if (found)
                    { 
                        #if defined ( GB_PHASE_1_OF_2 )
                        cjnz++ ;
                        #else
                        ASSERT (pC < pC_end) ;
                        Ci [pC] = i ;
                        GB_GETA (aij, Ax, pA) ;
                        GB_GETB (bij, Bx, pB) ;
                        GB_BINOP (GB_CX (pC), aij, bij) ;
                        pC++ ;
                        #endif
                    }
                }
                #if !defined ( GB_PHASE_1_OF_2 )
                ASSERT (pC == pC_end) ;
                #endif

            }
            else
            {

                //--------------------------------------------------------------
                // A(:,j) and B(:,j) have about the same # of entries
                //--------------------------------------------------------------

                // linear-time scan of A(:,j) and B(:,j)

                while (pA < pA_end && pB < pB_end)
                {
                    int64_t iA = Ai [pA] ;
                    int64_t iB = Bi [pB] ;
                    if (iA < iB)
                    { 
                        // A(i,j) exists but not B(i,j)
                        pA++ ;
                    }
                    else if (iB < iA)
                    { 
                        // B(i,j) exists but not A(i,j)
                        pB++ ;
                    }
                    else
                    { 
                        // both A(i,j) and B(i,j) exist
                        #if defined ( GB_PHASE_1_OF_2 )
                        cjnz++ ;
                        #else
                        ASSERT (pC < pC_end) ;
                        Ci [pC] = iB ;
                        GB_GETA (aij, Ax, pA) ;
                        GB_GETB (bij, Bx, pB) ;
                        GB_BINOP (GB_CX (pC), aij, bij) ;
                        pC++ ;
                        #endif
                        pA++ ;
                        pB++ ;
                    }
                }

                #if !defined ( GB_PHASE_1_OF_2 )
                ASSERT (pC == pC_end) ;
                #endif
            }

        }
        else
        {

            //------------------------------------------------------------------
            // M is present
            //------------------------------------------------------------------

            if (!Mask_comp && GB_IMIN (ajnz,bjnz) > 32 * mjnz)
            {

                //--------------------------------------------------------------
                // M(:,j) is very sparse, and not complemented
                //--------------------------------------------------------------

                // TODO: phase1: use a reduction on cjnz

                for ( ; pM < pM_end ; pM++)
                { 
                    // find M(i,j)
                    bool mij = false ;
                    int64_t i = Mi [pM] ;
                    cast_M (&mij, Mx +(pM*msize), 0) ;
                    if (!mij) continue ;

                    // find A(i,j)
                    int64_t pright = pA_end - 1 ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Ai, pA, pright, found) ;
                    if (!found) continue ;

                    // find B(i,j)
                    pright = pB_end - 1 ;
                    GB_BINARY_SEARCH (i, Bi, pB, pright, found) ;
                    if (!found) continue ;

                    // A(i,j) and B(i,j) found, and M(i,j) is true
                    #if defined ( GB_PHASE_1_OF_2 )
                    cjnz++ ;
                    #else
                    ASSERT (pC < pC_end) ;
                    Ci [pC] = i ;
                    GB_GETA (aij, Ax, pA) ;
                    GB_GETB (bij, Bx, pB) ;
                    GB_BINOP (GB_CX (pC), aij, bij) ;
                    pC++ ;
                    #endif
                }

            }
            else if (ajnz > 32 * bjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than B(:,j)
                //--------------------------------------------------------------

                // TODO: phase1: use a reduction on cjnz

                for ( ; pB < pB_end ; pB++)
                { 
                    int64_t i = Bi [pB] ;

                    // find A(i,j)
                    int64_t pright = pA_end - 1 ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Ai, pA, pright, found) ;
                    if (!found) continue ;

                    // find M(i,j)
                    bool mij = false ;
                    pright = pM_end - 1 ;
                    GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;
                    if (found)
                    { 
                        cast_M (&mij, Mx +(pM*msize), 0) ;
                    }
                    if (Mask_comp)
                    { 
                        mij = !mij ;
                    }
                    if (!mij) continue ;

                    // A(i,j) and B(i,j) found, and M(i,j) is true
                    #if defined ( GB_PHASE_1_OF_2 )
                    cjnz++ ;
                    #else
                    ASSERT (pC < pC_end) ;
                    Ci [pC] = i ;
                    GB_GETA (aij, Ax, pA) ;
                    GB_GETB (bij, Bx, pB) ;
                    GB_BINOP (GB_CX (pC), aij, bij) ;
                    pC++ ;
                    #endif
                }

            }
            else if (bjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // B(:,j) is much denser than A(:,j)
                //--------------------------------------------------------------

                // TODO: phase1: use a reduction on cjnz

                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t i = Ai [pA] ;

                    // find B(i,j)
                    int64_t pright = pB_end - 1 ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Bi, pB, pright, found) ;
                    if (!found) continue ;

                    // find M(i,j)
                    bool mij = false ;
                    pright = pM_end - 1 ;
                    GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;
                    if (found)
                    { 
                        cast_M (&mij, Mx +(pM*msize), 0) ;
                    }
                    if (Mask_comp)
                    { 
                        mij = !mij ;
                    }
                    if (!mij) continue ;

                    // A(i,j) and B(i,j) found, and M(i,j) is true
                    #if defined ( GB_PHASE_1_OF_2 )
                    cjnz++ ;
                    #else
                    ASSERT (pC < pC_end) ;
                    Ci [pC] = i ;
                    GB_GETA (aij, Ax, pA) ;
                    GB_GETB (bij, Bx, pB) ;
                    GB_BINOP (GB_CX (pC), aij, bij) ;
                    pC++ ;
                    #endif
                }

            }
            else
            {

                //--------------------------------------------------------------
                // A(:,j) and B(:,j) have about the same # of entries
                //--------------------------------------------------------------

                while (pA < pA_end && pB < pB_end)
                {
                    int64_t iA = Ai [pA] ;
                    int64_t iB = Bi [pB] ;
                    if (iA < iB)
                    { 
                        // A(i,j) exists but not B(i,j)
                        pA++ ;
                    }
                    else if (iB < iA)
                    { 
                        // B(i,j) exists but not A(i,j)
                        pB++ ;
                    }
                    else
                    { 
                        // both A(i,j) and B(i,j) exist.  Check the mask.
                        bool mij = false ;  // M(i,j) false if not present
                        int64_t pright = pM_end - 1 ;
                        bool found ;
                        GB_BINARY_SEARCH (iB, Mi, pM, pright, found) ;
                        if (found)
                        { 
                            cast_M (&mij, Mx +(pM*msize), 0) ;
                        }
                        if (Mask_comp)
                        { 
                            mij = !mij ;
                        }
                        if (mij)
                        { 
                            // A(i,j) and B(i,j) found, and M(i,j) is true
                            #if defined ( GB_PHASE_1_OF_2 )
                            cjnz++ ;
                            #else
                            ASSERT (pC < pC_end) ;
                            Ci [pC] = iB ;
                            GB_GETA (aij, Ax, pA) ;
                            GB_GETB (bij, Bx, pB) ;
                            GB_BINOP (GB_CX (pC), aij, bij) ;
                            pC++ ;
                            #endif
                        }
                        pA++ ;
                        pB++ ;
                    }
                }
            }

            #if !defined ( GB_PHASE_1_OF_2 )
            ASSERT (pC == pC_end) ;
            #endif
        }

        //----------------------------------------------------------------------
        // final count of nnz (C (:,j))
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_1_OF_2 )
        Cp [k] = cjnz ;
        #endif

    }
}

