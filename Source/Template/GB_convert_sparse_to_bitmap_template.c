//------------------------------------------------------------------------------
// GB_convert_sparse_to_bitmap_template: convert A from sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse or hypersparse.  Axnew and Ab have the same type as A,
// and represent a bitmap format.

{

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int64_t avlen = A->vlen ;

    #if defined ( GB_A_TYPE )
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_A_TYPE *restrict Axnew = (GB_A_TYPE *) Ax_new ;
    #endif

    #ifdef GB_JIT_KERNEL
    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;
    const int64_t nzombies = A->nzombies ;
    #endif

    //--------------------------------------------------------------------------
    // convert from sparse/hyper to bitmap
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,j) to be operated on by this task
            //------------------------------------------------------------------

            int64_t j = GBH_A (Ah, k) ;
//          int64_t pA_start, pA_end ;
//          GB_get_pA (&pA_start, &pA_end, tid, k,
//              kfirst, klast, pstart_Aslice, Ap, avlen) ;
            GB_GET_PA (pA_start, pA_end, tid, k,
                kfirst, klast, pstart_Aslice, Ap [k], Ap [k+1]) ;

            // the start of A(:,j) in the new bitmap
            int64_t pA_new = j * avlen ;

            //------------------------------------------------------------------
            // convert A(:,j) from sparse to bitmap
            //------------------------------------------------------------------

            if (nzombies == 0)
            {
                for (int64_t p = pA_start ; p < pA_end ; p++)
                { 
                    // A(i,j) has index i, value Ax [p]
                    int64_t i = Ai [p] ;
                    int64_t pnew = i + pA_new ;
                    // move A(i,j) to its new place in the bitmap
                    // Axnew [pnew] = Ax [p]
                    GB_COPY (Axnew, pnew, Ax, p) ;
                    Ab [pnew] = 1 ;
                }
            }
            else
            {
                for (int64_t p = pA_start ; p < pA_end ; p++)
                {
                    // A(i,j) has index i, value Ax [p]
                    int64_t i = Ai [p] ;
                    if (!GB_IS_ZOMBIE (i))
                    { 
                        int64_t pnew = i + pA_new ;
                        // move A(i,j) to its new place in the bitmap
                        // Axnew [pnew] = Ax [p]
                        GB_COPY (Axnew, pnew, Ax, p) ;
                        Ab [pnew] = 1 ;
                    }
                }
            }
        }
    }
}

#undef GB_A_TYPE

