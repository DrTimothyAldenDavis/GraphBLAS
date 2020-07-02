//------------------------------------------------------------------------------
// GB_AxB_dot2_nomask:  C=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{
    int ntasks = naslice * nbslice ;

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        int a_tid = tid / nbslice ;
        int b_tid = tid % nbslice ;

        //----------------------------------------------------------------------
        // get A
        //----------------------------------------------------------------------

        GrB_Matrix A = Aslice [a_tid] ;
        bool A_is_slice = A->is_slice ;
        const int64_t *GB_RESTRICT Ap = A->p ;
        const int64_t *GB_RESTRICT Ah = A->h ;
        const int64_t *GB_RESTRICT Ai = A->i ;
        int64_t anvec = A->nvec ;
        int64_t A_hfirst = A->hfirst ;

        #if defined ( GB_PHASE_1_OF_2 )
        int64_t *GB_RESTRICT C_count = C_counts [a_tid] ;
        #else
        int64_t *GB_RESTRICT C_count_start =
            (a_tid == 0) ?         NULL : C_counts [a_tid] ;
        int64_t *GB_RESTRICT C_count_end   =
            (a_tid == naslice-1) ? NULL : C_counts [a_tid+1] ;
        const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *)
            (A_is_pattern ? NULL : A->x) ;
        #endif

        //----------------------------------------------------------------------
        // C=A'*B via dot products
        //----------------------------------------------------------------------

        for (int64_t kB = B_slice [b_tid] ; kB < B_slice [b_tid+1] ; kB++)
        {

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            int64_t j = (Bh == NULL) ? kB : Bh [kB] ;
            int64_t pB_start = Bp [kB] ;
            int64_t pB_end = Bp [kB+1] ;

            int64_t bjnz = pB_end - pB_start ;
            // no work to do if B(:,j) is empty
            if (bjnz == 0) continue ;

            //------------------------------------------------------------------
            // phase 1 of 2: skip if B(:,j) is dense
            //------------------------------------------------------------------

            #if defined ( GB_PHASE_1_OF_2 )
            if (bjnz == bvlen)
            { 
                // C(i,j) is if A(:i) not empty
                C_count [kB] = A->nvec_nonempty ;
                continue ;
            }
            #endif

            //------------------------------------------------------------------
            // phase 2 of 2: get the range of entries in C(:,j) to compute
            //------------------------------------------------------------------

            #if defined ( GB_PHASE_2_OF_2 )
            // this thread computes Ci and Cx [cnz:cnz_last]
            int64_t cnz = Cp [kB] +
                ((C_count_start == NULL) ? 0 : C_count_start [kB]) ;
            int64_t cnz_last = (C_count_end == NULL) ?
                (Cp [kB+1] - 1) :
                (Cp [kB] + C_count_end [kB] - 1) ;
            if (cnz > cnz_last) continue ;
            #endif

            //------------------------------------------------------------------
            // C(:,j) = A'*B(:,j)
            //------------------------------------------------------------------

            // get the first and last index in B(:,j)
            int64_t ib_first = Bi [pB_start] ;
            int64_t ib_last  = Bi [pB_end-1] ;

            // for each vector A(:,i):
            for (int64_t kA = 0 ; kA < anvec ; kA++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                int64_t i ;
                if (A_is_slice)
                {
                    i = (Ah == NULL) ? (A_hfirst + kA) : Ah [kA] ;
                }
                else
                {
                    i = (Ah == NULL) ? kA : Ah [kA] ;
                }
                int64_t pA = Ap [kA] ;
                int64_t pA_end = Ap [kA+1] ;

                //--------------------------------------------------------------
                // C(i,j) = A(:,i)'*B(:,j)
                //--------------------------------------------------------------

                #include "GB_AxB_dot_cij.c"
            }
        }
    }
}

