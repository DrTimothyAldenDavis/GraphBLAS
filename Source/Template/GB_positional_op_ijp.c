//------------------------------------------------------------------------------
// GB_positional_op_ijp: C = positional_op (A), depending j
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// TODO: use this kernel for GrB_extractTuples, to create J array.

// A can be jumbled.  If A is jumbled, so is C.
// if A and C are bitmap, not all of Cx need to be written to, but it's faster
// just to write to all of it.  C->b is copied from A->b in the caller.

{

    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // Cx = positional_op (A)
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {

        // if kfirst > klast then task tid does no work at all
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;

        //----------------------------------------------------------------------
        // C(:,kfirst:klast) = op (A(:,kfirst:klast))
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) and Cx to be operated on by this task
            //------------------------------------------------------------------

            int64_t j = GBH_A (Ah, k) ;
            int64_t pA_start, pA_end ;
            GB_get_pA (&pA_start, &pA_end, tid, k,
                kfirst, klast, pstart_Aslice, Ap, avlen) ;

            //------------------------------------------------------------------
            // C(:,j) = op (A(:,j))
            //------------------------------------------------------------------

            for (int64_t p = pA_start ; p < pA_end ; p++)
            { 
                // Cx [p] = op (A (i,j))
                GB_APPLY (p) ;
            }
        }
    }
}

#undef GB_APPLY

