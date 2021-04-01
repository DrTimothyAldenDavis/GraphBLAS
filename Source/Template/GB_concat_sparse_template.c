//------------------------------------------------------------------------------
// GB_concat_sparse_template: concatenate a tile into a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get C and the tile A
    //--------------------------------------------------------------------------

    const GB_CTYPE *GB_RESTRICT Ax = (GB_CTYPE *) A->x ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;

    //--------------------------------------------------------------------------
    // copy the tile A into C
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(static)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            int64_t j = GBH (Ah, k) ;
            int64_t jC = cvstart + j ;
            int64_t pC_start = W [jC] ;
            int64_t pA_start, pA_end ;
//          GB_get_pA (&pA_start, &pA_end, tid, k,
//              kfirst, klast, pstart_Aslice, Ap, avlen) ;
            int64_t p0 = GBP (Ap, k, avlen) ;
            int64_t p1 = GBP (Ap, k+1, avlen) ;
            if (k == kfirst)
            { 
                // First vector for task tid; may only be partially owned.
                pA_start = pstart_Aslice [tid] ;
                pA_end   = GB_IMIN (p1, pstart_Aslice [tid+1]) ;
            }
            else if (k == klast)
            { 
                // Last vector for task tid; may only be partially owned.
                pA_start = p0 ;
                pA_end   = pstart_Aslice [tid+1] ;
            }
            else
            { 
                // task tid entirely owns this vector A(:,k).
                pA_start = p0 ;
                pA_end   = p1 ;
            }
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = GBI (Ai, pA, avlen) ;
                int64_t pC = pC_start + pA - p0 ;
                Ci [pC] = cistart + i ;
                // Cx [pC] = Ax [pA] ;
                GB_COPY (pC, pA) ;
            }
        }
    }
}

#undef GB_CTYPE

