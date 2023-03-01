//------------------------------------------------------------------------------
// GB_emult_03_template: C = A.*B when A is bitmap/full and B is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse, with the same sparsity structure as B.  No mask is present, or
// M is bitmap/full.  A is bitmap/full, and B is sparse/hyper.

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;
    const int64_t vlen = B->vlen ;

    const int8_t  *restrict Ab = A->b ;

    const int64_t *restrict kfirst_Bslice = B_ek_slicing ;
    const int64_t *restrict klast_Bslice  = B_ek_slicing + B_ntasks ;
    const int64_t *restrict pstart_Bslice = B_ek_slicing + B_ntasks * 2 ;

    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;

    #ifdef GB_ISO_EMULT
    ASSERT (C->iso) ;
    #else
    ASSERT (!C->iso) ;
    ASSERT (!(A_iso && B_iso)) ;    // one of A or B can be iso, but not both
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    const int64_t  *restrict Cp = C->p ;
          int64_t  *restrict Ci = C->i ;

    //--------------------------------------------------------------------------
    // C=A.*B or C<#M>=A.*B
    //--------------------------------------------------------------------------

    if (M == NULL)
    {

        //----------------------------------------------------------------------
        // C = A.*B
        //----------------------------------------------------------------------

        if (GB_IS_BITMAP (A))
        {

            //------------------------------------------------------------------
            // Method3(a): C=A.*B where A is bitmap and B is sparse/hyper
            //------------------------------------------------------------------

            int tid ;
            #pragma omp parallel for num_threads(B_nthreads) schedule(dynamic,1)
            for (tid = 0 ; tid < B_ntasks ; tid++)
            {
                int64_t kfirst = kfirst_Bslice [tid] ;
                int64_t klast  = klast_Bslice  [tid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    int64_t j = GBH_B (Bh, k) ;
                    int64_t pA_start = j * vlen ;
                    int64_t pB, pB_end, pC ;
                    GB_get_pA_and_pC (&pB, &pB_end, &pC, tid, k, kfirst, klast,
                        pstart_Bslice, Cp_kfirst, Cp, vlen, Bp, vlen) ;
                    for ( ; pB < pB_end ; pB++)
                    { 
                        int64_t i = Bi [pB] ;
                        int64_t pA = pA_start + i ;
                        if (!Ab [pA]) continue ;
                        // C (i,j) = A (i,j) .* B (i,j)
                        Ci [pC] = i ;
                        #ifndef GB_ISO_EMULT
                        GB_DECLAREA (aij) ;
                        GB_GETA (aij, Ax, pA, A_iso) ;     
                        GB_DECLAREB (bij) ;
                        GB_GETB (bij, Bx, pB, B_iso) ;
                        GB_BINOP (GB_CX (pC), aij, bij, i, j) ;
                        #endif
                        pC++ ;
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // Method3(b): C=A.*B where A is full and B is sparse/hyper
            //------------------------------------------------------------------

            int tid ;
            #pragma omp parallel for num_threads(B_nthreads) schedule(dynamic,1)
            for (tid = 0 ; tid < B_ntasks ; tid++)
            {
                int64_t kfirst = kfirst_Bslice [tid] ;
                int64_t klast  = klast_Bslice  [tid] ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    int64_t j = GBH_B (Bh, k) ;
                    int64_t pA_start = j * vlen ;
                    int64_t pB, pB_end ;
                    GB_get_pA (&pB, &pB_end, tid, k, kfirst, klast,
                        pstart_Bslice, Bp, vlen) ;
                    for ( ; pB < pB_end ; pB++)
                    { 
                        // C (i,j) = A (i,j) .* B (i,j)
                        int64_t i = Bi [pB] ;
                        int64_t pA = pA_start + i ;
                        // Ci [pB] = i ; already defined
                        #ifndef GB_ISO_EMULT
                        GB_DECLAREA (aij) ;
                        GB_GETA (aij, Ax, pA, A_iso) ;
                        GB_DECLAREB (bij) ;
                        GB_GETB (bij, Bx, pB, B_iso) ;
                        GB_BINOP (GB_CX (pB), aij, bij, i, j) ;
                        #endif
                    }
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // Method3(c): C<#M>=A.*B; M and A are bitmap/full; B is sparse/hyper
        //----------------------------------------------------------------------

        const int8_t  *restrict Mb = M->b ;
        const GB_M_TYPE *restrict Mx = (Mask_struct) ? NULL : ((GB_M_TYPE *) M->x) ;
        const size_t msize = M->type->size ;

        int tid ;
        #pragma omp parallel for num_threads(B_nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < B_ntasks ; tid++)
        {
            int64_t kfirst = kfirst_Bslice [tid] ;
            int64_t klast  = klast_Bslice  [tid] ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                int64_t j = GBH_B (Bh, k) ;
                int64_t pA_start = j * vlen ;
                int64_t pB, pB_end, pC ;
                GB_get_pA_and_pC (&pB, &pB_end, &pC, tid, k, kfirst, klast,
                    pstart_Bslice, Cp_kfirst, Cp, vlen, Bp, vlen) ;
                for ( ; pB < pB_end ; pB++)
                { 
                    int64_t i = Bi [pB] ;
                    int64_t pA = pA_start + i ;
                    if (!GBB_A (Ab, pA)) continue ;
                    bool mij = GBB_M (Mb, pA) && GB_MCAST (Mx, pA, msize) ;
                    mij = mij ^ Mask_comp ;
                    if (!mij) continue ;
                    // C (i,j) = A (i,j) .* B (i,j)
                    Ci [pC] = i ;
                    #ifndef GB_ISO_EMULT
                    GB_DECLAREA (aij) ;
                    GB_GETA (aij, Ax, pA, A_iso) ;     
                    GB_DECLAREB (bij) ;
                    GB_GETB (bij, Bx, pB, B_iso) ;
                    GB_BINOP (GB_CX (pC), aij, bij, i, j) ;
                    #endif
                    pC++ ;
                }
            }
        }
    }
}

