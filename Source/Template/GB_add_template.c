//------------------------------------------------------------------------------
// GB_add_template:  phase1 and phase2 for C=A+B, C<M>=A+B, and C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

// PARALLEL: done, except when # threads > # vectors (as in GrB_Vector).
// all vectors of a GrB_Matrix C are computed fully in parallel.  A
// single GrB_Vector will use only one thread, however.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, M, and C
    //--------------------------------------------------------------------------

    const int64_t  *restrict Ap = A->p ;
    const int64_t  *restrict Ai = A->i ;
    int64_t vlen = A->vlen ;

    const int64_t  *restrict Bp = B->p ;
    const int64_t  *restrict Bi = B->i ;

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

    #if defined ( GB_PHASE_2_OF_2 )
    const GB_ATYPE *restrict Ax = A->x ;
    const GB_ATYPE *restrict Bx = B->x ;
    const int64_t  *restrict Cp = C->p ;
    const int64_t  *restrict Ch = C->h ;
          int64_t  *restrict Ci = C->i ;
          GB_CTYPE *restrict Cx = C->x ;
    int64_t Cnvec = C->nvec ;
    #endif

    //--------------------------------------------------------------------------
    // phase1: count entries in each C(:j); phase 2: compute C
    //--------------------------------------------------------------------------

    int nth = GB_IMIN (Cnvec, nthreads) ;
    #pragma omp parallel for num_threads(nth)
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
        int64_t kA = (C_to_A == NULL) ? j : C_to_A [k] ;
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
        int64_t kB = (C_to_B == NULL) ? j : C_to_B [k] ;
        if (kB >= 0)
        { 
            pB     = Bp [kB] ;
            pB_end = Bp [kB+1] ;
        }
        int64_t bjnz = pB_end - pB ;    // nnz (B (:,j))

        //----------------------------------------------------------------------
        // get M(:,j)
        //----------------------------------------------------------------------

        // TODO: if A==M or B==M is aliased, then no need to do GB_lookup

        int64_t pM = -1 ;
        int64_t pM_end = -1 ;
        if (Ch_is_Mh)
        { 
            // Ch is the same as M->h, so binary search is not needed
            ASSERT (Ch != NULL && Mh != NULL && Ch [k] == Mh [k]) ;
            pM     = Mp [k] ;
            pM_end = Mp [k+1] ;
        }
        else if (M != NULL)
        { 
            int64_t kM = 0 ;
            GB_lookup (M_is_hyper, Mh, Mp, &kM, Mnvec-1, j, &pM, &pM_end) ;
        }
        int64_t mjnz = pM_end - pM ;    // nnz (M (:,j))

        //----------------------------------------------------------------------
        // phase1: count nnz (C (:,j)); phase2: compute C(:,j)
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
        else 

        #endif

        if (M == NULL || (M != NULL && mjnz == 0 && Mask_comp))
        {

            //------------------------------------------------------------------
            // No mask, or M(:,j) is empty and complemented
            //------------------------------------------------------------------

            // if present, M(:,j) is ignored since !M(:,j) is all true

            #if defined ( GB_PHASE_1_OF_2 )

            if (A_and_B_are_disjoint)
            { 

                // only used by GB_wait, which computes A+T where T is the
                // matrix of pending tuples for A.  The pattern of pending
                // tuples is always disjoint with the pattern of A.
                cjnz = ajnz + bjnz ;

            }
            else

            #endif

            if (ajnz == vlen && bjnz == vlen)
            {

                //--------------------------------------------------------------
                // A(:,j) and B(:,j) dense
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = vlen ;
                #else
                ASSERT (cjnz == vlen) ;
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
                cjnz = vlen ;
                #else
                ASSERT (cjnz == vlen) ;
                for (int64_t i = 0 ; i < vlen ; i++)
                { 
                    Ci [pC + i] = i ;
                    GB_COPY_A_TO_C (GB_CX (pC + i), Ax, pA + i) ;
                }
                for (int64_t p = 0 ; p < bjnz ; p++)
                { 
                    int64_t i = Bi [pB + p] ;
                    GB_GETA (aij, Ax, pA + i) ;
                    GB_GETB (bij, Bx, pB + p) ;
                    GB_BINOP (GB_CX (pC + i), aij, bij) ;
                }
                #endif

            }
            else if (bjnz == vlen)
            {

                //--------------------------------------------------------------
                // A(:,j) is sparse, B(:,j) is dense
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = vlen ;
                #else
                ASSERT (cjnz == vlen) ;
                for (int64_t i = 0 ; i < vlen ; i++)
                { 
                    Ci [pC + i] = i ;
                    GB_COPY_B_TO_C (GB_CX (pC + i), Bx, pB + i) ;
                }
                for (int64_t p = 0 ; p < ajnz ; p++)
                { 
                    int64_t i = Ai [pA + p] ;
                    GB_GETA (aij, Ax, pA + p) ;
                    GB_GETB (bij, Bx, pB + i) ;
                    GB_BINOP (GB_CX (pC + i), aij, bij) ;
                }
                #endif

            }
            else if (ajnz == 0)
            {

                //--------------------------------------------------------------
                // A(:,j) is empty
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = bjnz ;
                #else
                ASSERT (cjnz == bjnz) ;
                for (int64_t p = 0 ; p < bjnz ; p++)
                { 
                    Ci [pC + p] = Bi [pB + p] ;
                    GB_COPY_B_TO_C (GB_CX (pC + p), Bx, pB + p) ;
                }
                #endif

            }
            else if (bjnz == 0)
            {

                //--------------------------------------------------------------
                // B(:,j) is empty
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = ajnz ;
                #else
                ASSERT (cjnz == ajnz) ;
                for (int64_t p = 0 ; p < ajnz ; p++)
                { 
                    Ci [pC + p] = Ai [pA + p] ;
                    GB_COPY_A_TO_C (GB_CX (pC + p), Ax, pA + p) ;
                }
                #endif

            }
            else if (Ai [pA_end-1] < Bi [pB])
            {

                //--------------------------------------------------------------
                // last entry of A(:,j) comes before first entry of B(:,j)
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = ajnz + bjnz ;
                #else
                ASSERT (cjnz == ajnz + bjnz) ;
                for (int64_t p = 0 ; p < ajnz ; p++)
                { 
                    Ci [pC + p] = Ai [pA + p] ;
                    GB_COPY_A_TO_C (GB_CX (pC + p), Ax, pA + p) ;
                }
                pC += ajnz ;
                for (int64_t p = 0 ; p < bjnz ; p++)
                { 
                    Ci [pC + p] = Bi [pB + p] ;
                    GB_COPY_B_TO_C (GB_CX (pC + p), Bx, pB + p) ;
                }
                #endif

            }
            else if (Bi [pB_end-1] < Ai [pA])
            {

                //--------------------------------------------------------------
                // last entry of B(:,j) comes before first entry of A(:,j)
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz = ajnz + bjnz ;
                #else
                ASSERT (cjnz == ajnz + bjnz) ;
                for (int64_t p = 0 ; p < bjnz ; p++)
                { 
                    Ci [pC + p] = Bi [pB + p] ;
                    GB_COPY_B_TO_C (GB_CX (pC + p), Bx, pB + p) ;
                }
                pC += bjnz ;
                for (int64_t p = 0 ; p < ajnz ; p++)
                { 
                    Ci [pC + p] = Ai [pA + p] ;
                    GB_COPY_A_TO_C (GB_CX (pC + p), Ax, pA + p) ;
                }
                #endif

            }

            #if defined ( GB_PHASE_1_OF_2 )
            else if (ajnz > 32 * bjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than B(:,j)
                //--------------------------------------------------------------

                // cjnz = ajnz + bjnz - nnz in the intersection

                cjnz = ajnz + bjnz ;
                for ( ; pB < pB_end ; pB++)
                { 
                    int64_t i = Bi [pB] ;
                    // find i in A(:,j)
                    int64_t pright = pA_end ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Ai, pA, pright, found) ;
                    if (found) cjnz-- ;
                }

            }
            else if (bjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // B(:,j) is must denser than A(:,j)
                //--------------------------------------------------------------

                // cjnz = ajnz + bjnz - nnz in the intersection

                cjnz = ajnz + bjnz ;
                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t i = Ai [pA] ;
                    // find i in B(:,j)
                    int64_t pright = pB_end ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Bi, pB, pright, found) ;
                    if (found) cjnz-- ;
                }

            }
            #endif

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
                        // C (iA,j) = A (iA,j)
                        #if defined ( GB_PHASE_2_OF_2 )
                        Ci [pC] = iA ;
                        GB_COPY_A_TO_C (GB_CX (pC), Ax, pA) ;
                        #endif
                        pA++ ;
                    }
                    else if (iA > iB)
                    { 
                        // C (iB,j) = B (iB,j)
                        #if defined ( GB_PHASE_2_OF_2 )
                        Ci [pC] = iB ;
                        GB_COPY_B_TO_C (GB_CX (pC), Bx, pB) ;
                        #endif
                        pB++ ;
                    }
                    else
                    { 
                        // C (i,j) = A (i,j) + B (i,j)
                        #if defined ( GB_PHASE_2_OF_2 )
                        Ci [pC] = iB ;
                        GB_GETA (aij, Ax, pA) ;
                        GB_GETB (bij, Bx, pB) ;
                        GB_BINOP (GB_CX (pC), aij, bij) ;
                        #endif
                        pA++ ;
                        pB++ ;
                    }
                    #if defined ( GB_PHASE_2_OF_2 )
                    pC++ ;
                    #else
                    cjnz++ ;
                    #endif
                }

                //--------------------------------------------------------------
                // A (:,j) or B (:,j) have entries left; not both
                //--------------------------------------------------------------

                #if defined ( GB_PHASE_1_OF_2 )
                cjnz += (pA_end - pA) + (pB_end - pB) ;
                #else
                for ( ; pA < pA_end ; pA++, pC++)
                { 
                    // C (i,j) = A (i,j)
                    Ci [pC] = Ai [pA] ;
                    GB_COPY_A_TO_C (GB_CX (pC), Ax, pA) ;
                }
                for ( ; pB < pB_end ; pB++, pC++)
                { 
                    // C (i,j) = B (i,j)
                    Ci [pC] = Bi [pB] ;
                    GB_COPY_B_TO_C (GB_CX (pC), Bx, pB) ;
                }
                ASSERT (pC == pC_end) ;
                #endif
            }

        }
        else
        {

            //------------------------------------------------------------------
            // M is present
            //------------------------------------------------------------------

            while (pA < pA_end || pB < pB_end)
            {

                //--------------------------------------------------------------
                // get the next i for A(:,j) + B (:,j)
                //--------------------------------------------------------------

                int64_t iA = (pA < pA_end) ? Ai [pA] : INT64_MAX ;
                int64_t iB = (pB < pB_end) ? Bi [pB] : INT64_MAX ;
                int64_t i = GB_IMIN (iA, iB) ;

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                bool mij = false ;  // M(i,j) false if not present
                int64_t pright = pM_end - 1 ;
                bool found ;
                GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;
                if (found)
                { 
                    cast_M (&mij, Mx +(pM*msize), 0) ;
                }
                if (Mask_comp)
                { 
                    mij = !mij ;
                }

                //--------------------------------------------------------------
                // C(i,j) = A(i,j), B(i,j), or A(i,j) + B(i,j) via M(i,j)
                //--------------------------------------------------------------

                if (iA < iB)
                {
                    if (mij)
                    { 
                        // C (i,j) = A (i,j)
                        #if defined ( GB_PHASE_1_OF_2 )
                        cjnz++ ;
                        #else
                        Ci [pC] = i ;
                        GB_COPY_A_TO_C (GB_CX (pC), Ax, pA) ;
                        pC++ ;
                        #endif
                    }
                    pA++ ;
                }
                else if (iA > iB)
                {
                    if (mij)
                    { 
                        // C (i,j) = B (i,j)
                        #if defined ( GB_PHASE_1_OF_2 )
                        cjnz++ ;
                        #else
                        Ci [pC] = i ;
                        GB_COPY_B_TO_C (GB_CX (pC), Bx, pB) ;
                        pC++ ;
                        #endif
                    }
                    pB++ ;
                }
                else
                {
                    if (mij)
                    { 
                        // C (i,j) = A (i,j) + B (i,j)
                        #if defined ( GB_PHASE_1_OF_2 )
                        cjnz++ ;
                        #else
                        Ci [pC] = i ;
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

            #if defined ( GB_PHASE_2_OF_2 )
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

