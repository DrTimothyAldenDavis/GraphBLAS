//------------------------------------------------------------------------------
// GB_add_template:  C=A+B, C<M>=A+B, and C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

// PARALLEL:  all vectors of a GrB_Matrix C are computed fully in parallel.  A
// single GrB_Vector will use only one thread, however.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, M, and C
    //--------------------------------------------------------------------------

    const int64_t  *restrict Ap = A->p ;
    const int64_t  *restrict Ai = A->i ;
    const GB_ATYPE *restrict Ax = A->x ;

    const int64_t  *restrict Bp = B->p ;
    const int64_t  *restrict Bi = B->i ;
    const GB_ATYPE *restrict Bx = B->x ;

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

    const int64_t  *restrict Cp = C->p ;
    const int64_t  *restrict Ch = C->h ;
          int64_t  *restrict Ci = C->i ;
          GB_CTYPE *restrict Cx = C->x ;
    int64_t Cnvec = C->nvec ;

    //--------------------------------------------------------------------------
    // C=A+B, C<M>=A+B, or C<!M>=A+B
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads)
    for (int64_t k = 0 ; k < Cnvec ; k++)
    {

        //----------------------------------------------------------------------
        // get j, the kth vector of C
        //----------------------------------------------------------------------

        int64_t j = (Ch == NULL) ? k : Ch [k] ;
        int64_t pC     = Cp [k] ;
        int64_t pC_end = Cp [k+1] ;
        int64_t cjnz = pC_end - pC ;
        if (cjnz == 0) continue ;

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
        // compute C(:,j)
        //----------------------------------------------------------------------

        if (M == NULL || (M != NULL && mjnz == 0 && Mask_comp))
        {

            //------------------------------------------------------------------
            // No mask, or M(:,j) is empty and complemented
            //------------------------------------------------------------------

            for ( ; pA < pA_end && pB < pB_end ; pC++)
            {

                //--------------------------------------------------------------
                // C(i,j) = A(i,j), B(i,j), or A(i,j) + B(i,j)
                //--------------------------------------------------------------

                int64_t iA = Ai [pA] ;
                int64_t iB = Bi [pB] ;
                if (iA < iB)
                { 
                    // C (iA,j) = A (iA,j)
                    Ci [pC] = iA ;
                    GB_COPY_A_TO_C (GB_CX (pC), Ax, pA) ;
                    pA++ ;
                }
                else if (iA > iB)
                { 
                    // C (iB,j) = B (iB,j)
                    Ci [pC] = iB ;
                    GB_COPY_B_TO_C (GB_CX (pC), Bx, pB) ;
                    pB++ ;
                }
                else
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    Ci [pC] = iB ;
                    GB_GETA (aij, Ax, pA) ;
                    GB_GETB (bij, Bx, pB) ;
                    GB_BINOP (GB_CX (pC), aij, bij) ;
                    pA++ ;
                    pB++ ;
                }
            }

            //------------------------------------------------------------------
            // A (:,j) or B (:,j) have entries left; not both
            //------------------------------------------------------------------

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
                        Ci [pC] = i ;
                        GB_COPY_A_TO_C (GB_CX (pC), Ax, pA) ;
                        pC++ ;
                    }
                    pA++ ;
                }
                else if (iA > iB)
                {
                    if (mij)
                    { 
                        // C (i,j) = B (i,j)
                        Ci [pC] = i ;
                        GB_COPY_B_TO_C (GB_CX (pC), Bx, pB) ;
                        pC++ ;
                    }
                    pB++ ;
                }
                else
                {
                    if (mij)
                    { 
                        // C (i,j) = A (i,j) + B (i,j)
                        Ci [pC] = i ;
                        GB_GETA (aij, Ax, pA) ;
                        GB_GETB (bij, Bx, pB) ;
                        GB_BINOP (GB_CX (pC), aij, bij) ;
                        pC++ ;
                    }
                    pA++ ;
                    pB++ ;
                }
            }
        }

        ASSERT (pC == pC_end) ;
    }
}

