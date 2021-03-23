//------------------------------------------------------------------------------
// GB_AxB_saxpy4_mask_template: gather/scatter the mask M from/to Wf
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The mask M is sparse or hypersparse, and needs to be scattered into Wf,
// or cleared from Wf.

{

    ASSERT (M_is_sparse_or_hyper && apply_mask) ;
    #ifdef GB_MTYPE
    const GB_MTYPE *GB_RESTRICT Mx = (const GB_MTYPE *) M->x ;
    #endif
    int taskid ;

    if (C_and_B_are_hyper)
    {
        #pragma omp parallel for num_threads(M_nthreads) schedule(static)
        for (taskid = 0 ; taskid < M_ntasks ; taskid++)
        {
            int64_t kfirst = kfirst_Mslice [taskid] ;
            int64_t klast  = klast_Mslice  [taskid] ;
            int64_t ck = 0 ;
            for (int64_t kk = kfirst ; kk <= klast ; kk++)
            {
                // scatter M(:,j) into Wf
                int64_t j = GBH (Mh, kk) ;
                int64_t pM_start, pM_end ;
                GB_get_pA (&pM_start, &pM_end, taskid, kk,
                    kfirst, klast, pstart_Mslice, Mp, mvlen) ;
                bool found ;
                // look for j in Ch
                int64_t pright = cnvec ;
                GB_BINARY_SEARCH (j, Ch, ck, pright, found) ;
                if (found)
                {
                    // C(:,j) is the (ck)th vector in C
                    int8_t *GB_RESTRICT Hf = Wf + ck * cvlen ;
                    GB_PRAGMA_SIMD
                    for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                    {
                        Hf [Mi [pM]] = GB_MASK_ij (pM) ;
                    }
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(M_nthreads) schedule(static)
        for (taskid = 0 ; taskid < M_ntasks ; taskid++)
        {
            int64_t kfirst = kfirst_Mslice [taskid] ;
            int64_t klast  = klast_Mslice  [taskid] ;
            for (int64_t kk = kfirst ; kk <= klast ; kk++)
            {
                // scatter M(:,j) into Wf
                int64_t j = GBH (Mh, kk) ;
                int64_t pM_start, pM_end ;
                GB_get_pA (&pM_start, &pM_end, taskid, kk,
                    kfirst, klast, pstart_Mslice, Mp, mvlen) ;
                // C(:,j) is the jth vector in C
                int8_t *GB_RESTRICT Hf = Wf + j * cvlen ;
                GB_PRAGMA_SIMD
                for (int64_t pM = pM_start ; pM < pM_end ; pM++)
                {
                    Hf [Mi [pM]] = GB_MASK_ij (pM) ;
                }
            }
        }
    }
}

#undef GB_MTYPE
#undef GB_MASK_ij
