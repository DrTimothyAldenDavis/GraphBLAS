//------------------------------------------------------------------------------
// GB_AxB_dot2_template:  C=A'B, C<!M>=A'*B, or C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// TODO: rename GB_bitmap_AxB_dot_template.c

// A and B are sparse, bitmap, or full; never hypersparse.  If the input
// matrices A and/or B are hypersparse, they are packed into sparse matrices,
// and C is unpacked from bitmap to sparse/hypersparse when done.

{

    //--------------------------------------------------------------------------
    // C=A'*B, C<M>=A'*B, or C<!M>=A'*B where C is bitmap
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
        reduction(+:cnvals)
    for (tid = 0 ; tid < ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // get the task descriptor
        //----------------------------------------------------------------------

        const int a_tid = tid / nbslice ;
        const int b_tid = tid % nbslice ;
        const int64_t kA_start = A_slice [a_tid] ;
        const int64_t kA_end   = A_slice [a_tid+1] ;
        const int64_t kB_start = B_slice [b_tid] ;
        const int64_t kB_end   = B_slice [b_tid+1] ;

        //----------------------------------------------------------------------
        // C=A'*B, C<M>=A'*B, or C<!M>=A'*B via dot products
        //----------------------------------------------------------------------

        for (int64_t j = kB_start ; j < kB_end ; j++)
        {

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            #if GB_B_IS_SPARSE_OR_HYPER
            // B is sparse (never hypersparse)
            const int64_t pB_start = Bp [j] ;
            const int64_t pB_end   = Bp [j+1] ;
            const int64_t bjnz = pB_end - pB_start ;
                #if ( GB_A_IS_SPARSE_OR_HYPER )
                // get the first and last index in B(:,j)
                const int64_t ib_first = Bi [pB_start] ;
                const int64_t ib_last  = Bi [pB_end-1] ;
                #endif
            #else
            // B is bitmap or full
            const int64_t pB_start = j * vlen ;
            const int64_t bjnz = vlen ;
            #endif
            // no work to do if B(:,j) is empty
            if (bjnz == 0) continue ;

            //------------------------------------------------------------------
            // get C(:,j)
            //------------------------------------------------------------------

            const int64_t pC_start = j * cvlen ;

            //------------------------------------------------------------------
            // get M(:,j), if present
            //------------------------------------------------------------------

            #if defined ( GB_ANY_SPECIALIZED )
            // M is bitmap, and pM is the same as pC_start
            #elif defined ( GB_MASK_IS_PRESENT )
            // TODO: delete this and scatter M into the C bitmap if sparse,
            // or use in-place is M is dense, bitmap, or full
            // find vector j in M
            int64_t pM, pM_end ;
            bool mdense = false ;           // TODO remove this
            if (!M_is_bitmap_or_full)
            {
                // M is hypersparse or sparse
                int64_t mpleft = 0 ;
                GB_lookup (M_is_hyper, Mh, Mp, mvlen, &mpleft, mnvec-1, j,
                    &pM, &pM_end) ;
                int64_t mjnz = pM_end - pM ;
                mdense = (mjnz == mvlen) ;  // TODO remove this
            }
            #endif

            //------------------------------------------------------------------
            // C(:,j)<#M(:,j)> = A'*B(:,j), or C(:,j) = A'*B(:,j) if no mask
            //------------------------------------------------------------------

            for (int64_t i = kA_start ; i < kA_end ; i++)
            {

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                #if defined ( GB_ANY_SPECIALIZED )
                // M is bitmap and structural; Mask_comp true
                if (!Mb [pC_start + i])
                #elif defined ( GB_MASK_IS_PRESENT )
                bool mij ;
                if (M_is_bitmap)
                {
                    // M is bitmap
                    mij = Mb [pC_start + i] &&
                          GB_mcast (Mx, pC_start + i, msize) ;
                }
                else if (M_is_full || mdense)
                {
                    // M is full
                    mij = GB_mcast (Mx, pC_start + i, msize) ;
                }
                else if (mdense)
                { 
                    // M sparse/hyper, with a fully-populated vector M(:,j)
                    mij = GB_mcast (Mx, pM + i, msize) ;
                }
                else
                {
                    // M(:,j) is sparse:
                    // TODO: delete this and scatter M into the C bitmap
                    // instead.
                    bool found ;
                    int64_t pright = pM_end - 1 ;
                    GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;
                    mij = found && GB_mcast (Mx, pM, msize) ;
                }
                if (mij ^ Mask_comp)
                #endif
                { 

                    //----------------------------------------------------------
                    // C(i,j) = A(:,i)'*B(:,j)
                    //----------------------------------------------------------

                    #if GB_A_IS_SPARSE_OR_HYPER
                    int64_t pA = Ap [i] ;
                    const int64_t pA_end = Ap [i+1] ;
                    int64_t ainz = pA_end - pA ;
                    if (ainz == 0) continue ;
                    #else
                    const int64_t pA = i * vlen ;
                    #endif
                    #include "GB_AxB_dot2_cij.c"
                }
            }
        }
    }
}

#undef GB_A_IS_SPARSE_OR_HYPER
#undef GB_A_IS_BITMAP
#undef GB_A_IS_FULL
#undef GB_B_IS_SPARSE_OR_HYPER
#undef GB_B_IS_BITMAP
#undef GB_B_IS_FULL

