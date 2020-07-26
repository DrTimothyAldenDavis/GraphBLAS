//------------------------------------------------------------------------------
// GB_positional_op_ip: C = positional_op (A), depending only on i
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // Cx = positional_op (A)
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        int64_t pstart, pend ;
        GB_PARTITION (pstart, pend, anz, tid, ntasks) ;
        GB_PRAGMA_SIMD
        for (int64_t p = pstart ; p < pend ; p++)
        { 
            // GB_POSITION is either i or i+1
            int64_t i = GBI (Ai, p, avlen) ;
            Cx_int [p] = GB_POSITION ;
        }
    }
}

#undef GB_POSITION

