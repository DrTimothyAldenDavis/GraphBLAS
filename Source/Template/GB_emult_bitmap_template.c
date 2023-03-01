//------------------------------------------------------------------------------
// GB_emult_bitmap_template: C = A.*B, C<M>=A.*B, and C<!M>=A.*B, C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap.  A and B are bitmap or full.  M depends on the method

{

    //--------------------------------------------------------------------------
    // get C, A, and B
    //--------------------------------------------------------------------------

    const int8_t  *restrict Ab = A->b ;
    const int8_t  *restrict Bb = B->b ;
    const int64_t vlen = A->vlen ;

    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A) || GB_as_if_full (A)) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (A) || GB_as_if_full (B)) ;

    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;

    int8_t *restrict Cb = C->b ;
    const int64_t cnz = GB_nnz_held (C) ;

    #ifdef GB_ISO_EMULT
    ASSERT (C->iso) ;
    #else
    ASSERT (!C->iso) ;
    ASSERT (!(A_iso && B_iso)) ;    // one of A or B can be iso, but not both
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    //--------------------------------------------------------------------------
    // C=A.*B, C<M>=A.*B, or C<!M>=A.*B: C is bitmap
    //--------------------------------------------------------------------------

    // TODO modify this method so it can modify C in-place, and also use the
    // accum operator.
    int64_t cnvals = 0 ;

    if (ewise_method == GB_EMULT_METHOD5)
    {
        // C=A.*B; C bitmap, M not present, A and B are bitmap/full
        #include "GB_emult_bitmap_5.c"
    }
    else if (ewise_method == GB_EMULT_METHOD6)
    {
        // C<!M>=A.*B; C bitmap, M sparse, A and B are bitmap/full
        #include "GB_emult_bitmap_6.c"
    }
    else // if (ewise_method == GB_EMULT_METHOD7)
    {
        // C<#M>=A.*B; C bitmap; M, A, and B are all bitmap/full
        #include "GB_emult_bitmap_7.c"
    }

    C->nvals = cnvals ;
}

