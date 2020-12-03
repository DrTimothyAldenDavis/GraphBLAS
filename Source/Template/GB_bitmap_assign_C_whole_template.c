//------------------------------------------------------------------------------
// GB_bitmap_assign_C_whole_template: iterate over a bitmap matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The #include'ing file defines a GB_CIJ_WORK macro for the body of the loop,
// which operates on the entry C(iC,jC) at position Cx [pC] and Cb [pC].  The C
// matrix held in bitmap form.  If the mask matrix is also a bitmap matrix or
// full matrix, the GB_GET_MIJ macro can compute the effective value of the
// mask for the C(iC,jC) entry.

// C must be bitmap or full.  If M is accessed, it must also be bitmap or full.

#ifndef GB_GET_MIJ
#define GB_GET_MIJ(mij,pM) ;
#endif

{
    // iterate over all of C(:,:).
    int64_t pC ;
    int nthreads = GB_nthreads (cnzmax, chunk, nthreads_max) ;
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(+:cnvals)
    for (pC = 0 ; pC < cnzmax ; pC++)
    { 
        // int64_t iC = pC % cvlen ;
        // int64_t jC = pC / cvlen ;
        GB_GET_MIJ (mij, pC) ;          // mij = Mask (pC)
        GB_CIJ_WORK (pC) ;              // operate on C(iC,jC)
    }
}

