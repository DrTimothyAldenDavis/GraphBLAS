//------------------------------------------------------------------------------
// GB_subassign_22_template: C += y where C is dense and y is a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    ASSERT (!C->iso) ;

    //--------------------------------------------------------------------------
    // C += y where C is dense and y is a scalar
    //--------------------------------------------------------------------------

    int64_t pC ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (pC = 0 ; pC < cnz ; pC++)
    { 
        // Cx [pC] += ywork
        GB_ACCUMULATE_scalar (Cx, pC, ywork) ;
    }
}

