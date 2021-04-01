//------------------------------------------------------------------------------
// GB_concat_bitmap_template: concatenate a tile into a bitmap matrix
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
    int8_t *GB_RESTRICT Cb = C->b ;

    //--------------------------------------------------------------------------
    // copy the tile A into C
    //--------------------------------------------------------------------------

    switch (sparsity)
    {

        case GxB_FULL : // A is full
        {
            int64_t pA ;
            #pragma omp for num_threads(nthreads) schedule(static)
            for (pA = 0 ; pA < anz ; pA++)
            {
                int64_t i = pA % vlen ;
                int64_t j = pA / vlen ;
                int64_t iC = cistart + i ;
                int64_t jC = cvstart + j ;
                int64_t pC = iC + jC * cvlen ;
                // Cx [pC] = Ax [pA] ;
                GB_COPY (pC, pA) ;
                Cb [pC] = 1 ;
            }
        }
        break ;

        case GxB_BITMAP : // A is bitmap
        {
            const int8_t *GB_RESTRICT Ab = A->b ;
            int64_t pA ;
            #pragma omp for num_threads(nthreads) schedule(static)
            for (pA = 0 ; pA < anz ; pA++)
            {
                if (Ab (pA))
                {
                    int64_t i = pA % vlen ;
                    int64_t j = pA / vlen ;
                    int64_t iC = cistart + i ;
                    int64_t jC = cvstart + j ;
                    int64_t pC = iC + jC * cvlen ;
                    // Cx [pC] = Ax [pA] ;
                    GB_COPY (pC, pA) ;
                    Cb [pC] = 1 ;
                }
            }
        }
        break ;

        default : // A is sparse or hypersparse
        {
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
                    int64_t pC_start = cistart + jC * cvlen ;
                    int64_t pA_start, pA_end ;
                    GB_get_pA (&pA_start, &pA_end, tid, k,
                        kfirst, klast, pstart_Aslice, Ap, avlen) ;
                    for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                    {
                        int64_t i = Ai [pA] ;
                        int64_t pC = pC_start + i ;
                        // Cx [pC] = Ax [pA] ;
                        GB_COPY (pC, pA) ;
                        Cb [pC] = 1 ;
                    }
                }
            }
            break ;
        }
    }
}

#undef GB_CTYPE

