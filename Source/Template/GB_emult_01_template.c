//------------------------------------------------------------------------------
// GB_emult_01_template: C = A.*B when A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse, with the same sparsity structure as A.  No mask is present.
// A is sparse/hyper, and B is bitmap/full.  This method also handles the case
// when the original input A is bitmap/full and B is sparse/hyper, by computing
// B.*A with the operator flipped.

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t vlen = A->vlen ;

    const int8_t  *GB_RESTRICT Bb = B->b ;

    const int64_t *GB_RESTRICT kfirst_Aslice = A_ek_slicing ;
    const int64_t *GB_RESTRICT klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *GB_RESTRICT pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    #if GB_FLIPPED
    const GB_BTYPE *GB_RESTRICT Ax = (GB_BTYPE *) A->x ;
    const GB_ATYPE *GB_RESTRICT Bx = (GB_ATYPE *) B->x ;
    #else
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) A->x ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) B->x ;
    #endif

    const int64_t  *GB_RESTRICT Cp = C->p ;
          int64_t  *GB_RESTRICT Ci = C->i ;
          GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;

    //--------------------------------------------------------------------------
    // C=A.*B
    //--------------------------------------------------------------------------

    if (GB_IS_BITMAP (B))
    {

        //----------------------------------------------------------------------
        // C=A.*B where A is sparse/hyper and B is bitmap
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < A_ntasks ; tid++)
        {
            int64_t kfirst = kfirst_Aslice [tid] ;
            int64_t klast  = klast_Aslice  [tid] ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                int64_t j = GBH (Ah, k) ;
                int64_t pB_start = j * vlen ;
                int64_t pA, pA_end, pC ;
                GB_get_pA_and_pC (&pA, &pA_end, &pC, tid, k, kfirst, klast,
                    pstart_Aslice, Cp_kfirst, Cp, vlen, Ap, vlen) ;
                for ( ; pA < pA_end ; pA++)
                {
                    int64_t i = Ai [pA] ;
                    int64_t pB = pB_start + i ;
                    if (!Bb [pB]) continue ;
                    // C (i,j) = A (i,j) .* B (i,j)
                    Ci [pC] = i ;
                    GB_GETA (aij, Ax, pA) ;     
                    GB_GETB (bij, Bx, pB) ;
                    #if GB_FLIPPED
                    GB_BINOP (GB_CX (pC), bij, aij, i, j) ;
                    #else
                    GB_BINOP (GB_CX (pC), aij, bij, i, j) ;
                    #endif
                    pC++ ;
                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C=A.*B where A is sparse/hyper and B is full
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
        for (tid = 0 ; tid < A_ntasks ; tid++)
        {
            int64_t kfirst = kfirst_Aslice [tid] ;
            int64_t klast  = klast_Aslice  [tid] ;
            for (int64_t k = kfirst ; k <= klast ; k++)
            {
                int64_t j = GBH (Ah, k) ;
                int64_t pB_start = j * vlen ;
                int64_t pA, pA_end ;
                GB_get_pA (&pA, &pA_end, tid, k, kfirst, klast,
                    pstart_Aslice, Ap, vlen) ;
                for ( ; pA < pA_end ; pA++)
                { 
                    // C (i,j) = A (i,j) .* B (i,j)
                    int64_t i = Ai [pA] ;
                    int64_t pB = pB_start + i ;
                    // Ci [pA] = i ; already defined
                    GB_GETA (aij, Ax, pA) ;
                    GB_GETB (bij, Bx, pB) ;
                    #if GB_FLIPPED
                    GB_BINOP (GB_CX (pA), bij, aij, i, j) ;
                    #else
                    GB_BINOP (GB_CX (pA), aij, bij, i, j) ;
                    #endif
                }
            }
        }
    }
}

