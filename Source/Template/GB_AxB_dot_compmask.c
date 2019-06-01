//------------------------------------------------------------------------------
// GB_AxB_dot_compmask:  C<!M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// TODO: delete this method entirely?  Use dot2 instead.

{

    //--------------------------------------------------------------------------
    // C<!M>=A'*B via dot products
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
        #endif

        //----------------------------------------------------------------------
        // get M(:,j)
        //----------------------------------------------------------------------

        // find vector j in M
        int64_t pM, pM_end ;
        GB_lookup (M_is_hyper, Mh, Mp, &mpleft, mpright, j, &pM, &pM_end) ;

        //----------------------------------------------------------------------
        // C(:,j)<!M(:,j)> = A'*B(:,j)
        //----------------------------------------------------------------------

        // get the first and last index in B(:,j)
        int64_t ib_first = Bi [pB_start] ;
        int64_t ib_last  = Bi [pB_end-1] ;

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
            if (!mij)
            { 
                // C(i,j) = A(:,i)'*B(:,j)
                #include "GB_AxB_dot_cij.c"
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

