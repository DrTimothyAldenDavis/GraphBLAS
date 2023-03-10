//------------------------------------------------------------------------------
// GB_select bitmap_template: C=select(A,thunk) if A is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  A is bitmap or as-if-full.

{
    int8_t *Ab = A->b ;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const size_t asize = A->type->size ;
    const int64_t anz = avlen * avdim ;
    int64_t p, cnvals = 0 ;
    #pragma omp parallel for num_threads(nthreads) schedule(static) \
        reduction(+:cnvals)
    for (p = 0 ; p < anz ; p++)
    { 
        int64_t i = p % avlen ;
        int64_t j = p / avlen ;
        #if defined ( GB_ENTRY_SELECTOR )
            // test the existence and value of A(i,j)
            // FIXME: this tests A(i,j) even if it doesn't exist
            GB_TEST_VALUE_OF_ENTRY (keep, p) ;
        #endif
        int8_t cb = GBB_A (Ab, p) &&
        #if defined ( GB_ENTRY_SELECTOR )
            keep ;
        #elif defined ( GB_TRIL_SELECTOR )
            (j-i <= ithunk) ;
        #elif defined ( GB_TRIU_SELECTOR )
            (j-i >= ithunk) ;
        #elif defined ( GB_DIAG_SELECTOR )
            (j-i == ithunk) ;
        #elif defined ( GB_OFFDIAG_SELECTOR )
            (j-i != ithunk) ;
        #elif defined ( GB_ROWINDEX_SELECTOR )
            (i+ithunk != 0) ;
        #elif defined ( GB_COLINDEX_SELECTOR )
            (j+ithunk != 0) ;
        #elif defined ( GB_COLLE_SELECTOR )
            (j <= ithunk) ;
        #elif defined ( GB_COLGT_SELECTOR )
            (j > ithunk) ;
        #elif defined ( GB_ROWLE_SELECTOR )
            (i <= ithunk) ;
        #elif defined ( GB_ROWGT_SELECTOR )
            (i > ithunk) ;
        #endif
        Cb [p] = cb ;
        cnvals += cb ;
    }
    (*cnvals_handle) = cnvals ;
}

#undef GB_TRIL_SELECTOR
#undef GB_TRIU_SELECTOR
#undef GB_DIAG_SELECTOR
#undef GB_OFFDIAG_SELECTOR
#undef GB_ROWINDEX_SELECTOR
#undef GB_COLINDEX_SELECTOR
#undef GB_COLLE_SELECTOR
#undef GB_COLGT_SELECTOR
#undef GB_ROWLE_SELECTOR
#undef GB_ROWGT_SELECTOR
