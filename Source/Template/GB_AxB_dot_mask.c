//------------------------------------------------------------------------------
// GB_AxB_dot_mask:  C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get first and last non-empty vector of A
    //--------------------------------------------------------------------------

    int64_t ia_first = -1, ia_last = -1 ;
    if (A_is_hyper)
    {
        // A is hypersparse or hyperslice
        if (anvec > 0)
        { 
            ia_first = Ah [0] ;
            ia_last  = Ah [anvec-1] ;
        }
    }
    else
    { 
        // A is standard sparse, or a slice.  For a standard matrix, A->hfirst
        // is zero and A->nvec = A->vdim, so ia_first and ia_last include the
        // whole matrix.
        ia_first = A->hfirst ;
        ia_last  = A->hfirst + anvec - 1 ;
    }

    //--------------------------------------------------------------------------
    // C<M>=A'*B via dot products
    //--------------------------------------------------------------------------

    GBI_for_each_vector (B)
    {

        //----------------------------------------------------------------------
        // get B(:,j)
        //----------------------------------------------------------------------

        GBI_jth_iteration (j, pB_start, pB_end) ;
        int64_t bjnz = pB_end - pB_start ;
        // no work to do if B(:,j) is empty
        if (bjnz == 0) continue ;

        //----------------------------------------------------------------------
        // phase 2 of 2: get the range of entries in C(:,j) to compute
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_2_OF_2 )
        // this thread computes Ci and Cx [cnz:cnz_last]
        int64_t cnz = Cp [Iter_k] +
            ((C_count_start == NULL) ? 0 : C_count_start [Iter_k]) ;
        int64_t cnz_last = (C_count_end == NULL) ?
            (Cp [Iter_k+1] - 1) : (Cp [Iter_k] + C_count_end [Iter_k] - 1) ;

        if (cnz > cnz_last) continue ;
        GB_CIJ_REACQUIRE (cij, cnz) ;
        #endif

        //----------------------------------------------------------------------
        // get M(:,j)
        //----------------------------------------------------------------------

        // find vector j in M
        int64_t pM, pM_end ;
        GB_lookup (M_is_hyper, Mh, Mp, &mpleft, mpright, j, &pM, &pM_end) ;
        // no work to do if M(:,j) is empty
        int64_t mjnz = pM_end - pM ;
        if (mjnz == 0) continue ;

        //----------------------------------------------------------------------
        // C(:,j)<M(:,j)> = A'*B(:,j)
        //----------------------------------------------------------------------

        // get the first and last index in B(:,j)
        int64_t ib_first = Bi [pB_start] ;
        int64_t ib_last  = Bi [pB_end-1] ;

        // get the first and last index in M(:,j)
        int64_t im_first = Mi [pM] ;
        int64_t im_last  = Mi [pM_end-1] ;

        // no work to do if M(:,j) does not include any vectors in A
        if (ia_last < im_first || im_last < ia_first) continue ;

        if (mjnz <= anvec)
        {

            //------------------------------------------------------------------
            // M(:,j) is sparser than the vectors of A 
            //------------------------------------------------------------------

            // advance pM to the first vector of A
            if (im_first < ia_first)
            {
                // search M(:,j) for the first vector of A
                int64_t pright = pM_end - 1 ;
                GB_BINARY_TRIM_SEARCH (ia_first, Mi, pM, pright) ;
            }

            int64_t pleft = 0 ;
            int64_t pright = anvec-1 ;

            // iterate over all entries in M(:,j)
            for ( ; pM < pM_end ; pM++)
            {

                // get the next entry M(i,j)
                int64_t i = Mi [pM] ;
                if (i > ia_last)
                {
                    // i is past last vector of A so the remainder of
                    // M(:,j) can be ignored
                    break ;
                }

                // the binary_trim_search of M(:,j), above, has trimmed the
                // leading part of M(:,j), so i >= ia_first must hold here.
                // The break statement above ensures that i <= ia_last holds
                // here.  So M(i,j) exists, and i is in the range of the
                // vectors of A
                ASSERT (i >= ia_first && i <= ia_last) ;

                // get the value of M(i,j) and skip if false
                bool mij ;
                cast_M (&mij, Mx +(pM*msize), 0) ;
                if (!mij) continue ;

                // get A(:,i), if it exists
                int64_t pA, pA_end ;
                if (A->is_slice && !A_is_hyper)
                {
                    // A is a slice
                    int64_t ka = i - ia_first ;
                    ASSERT (ka >= 0 && ka < anvec) ;
                    pA     = Ap [ka] ;
                    pA_end = Ap [ka+1] ;
                }
                else
                {
                    // A is sparse, hypersparse, or hyperslice
                    GB_lookup (A_is_hyper, Ah, Ap, &pleft, pright, i,
                        &pA, &pA_end) ;
                }

                // C(i,j) = A(:,i)'*B(:,j)
                #include "GB_AxB_dot_cij.c"
            }

        }
        else
        {

            //------------------------------------------------------------------
            // M(:,j) is denser than the vectors of A 
            //------------------------------------------------------------------

            // for each vector A(:,i):
            GBI_for_each_vector_with_iter (Iter_A, A)
            {
                GBI_jth_iteration_with_iter (Iter_A, i, pA, pA_end) ;

                // A(:,i) and B(:,j) are both present.  Check M(i,j).
                // TODO: skip binary search if mask is dense.
                bool mij = false ;
                bool found ;
                int64_t pright = pM_end - 1 ;
                GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;
                if (found)
                {
                    cast_M (&mij, Mx +(pM*msize), 0) ;
                }
                if (mij)
                { 
                    // C(i,j) = A(:,i)'*B(:,j)
                    #include "GB_AxB_dot_cij.c"
                }
            }
        }

        //----------------------------------------------------------------------
        // single phase: log the end of C(:,j)
        //----------------------------------------------------------------------

        #if defined ( GB_SINGLE_PHASE )
        // cannot fail since C->plen is at the upper bound: # of non-empty
        // columns of B
        info = GB_jappend (C, j, &jlast, cnz, &cnz_last, NULL) ;
        ASSERT (info == GrB_SUCCESS) ;
        #endif
    }

    //--------------------------------------------------------------------------
    // single phase: finalize C
    //--------------------------------------------------------------------------

    #if defined ( GB_SINGLE_PHASE )
    GB_jwrapup (C, jlast, cnz) ;
    #endif
}

