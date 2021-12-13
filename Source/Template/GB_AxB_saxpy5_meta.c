//------------------------------------------------------------------------------
// GB_AxB_saxpy5_meta.c: C+=A*B when C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method is only used for built-in semirings with no typecasting.
// The accumulator matches the semiring monoid.
// The ANY monoid is not supported.

// C is as-if-full.
// A is bitmap or full.
// B is sparse or hypersparse.

#if GB_IS_ANY_MONOID
#error "saxpy5 not defined for the ANY monoid"
#endif

{

    //--------------------------------------------------------------------------
    // get C, A, and B
    //--------------------------------------------------------------------------

    ASSERT (GB_as_if_full (C)) ;
    const int64_t m = C->vlen ;     // # of rows of C
    ASSERT (C->vlen == A->vlen) ;
    ASSERT (C->vdim == B->vdim) ;
    ASSERT (A->vdim == B->vlen) ;

    const int8_t *restrict Ab = A->b ;
    const bool A_iso = A->iso ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    ASSERT (A_is_bitmap || GB_as_if_full (A)) ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;
    const bool B_iso = B->iso ;
    const int64_t bnvec = B->nvec ;
    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    ASSERT (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B)) ;

    #if !GB_A_IS_PATTERN
    const GB_ATYPE *restrict Ax = (GB_ATYPE *) A->x ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_BTYPE *restrict Bx = (GB_BTYPE *) B->x ;
    #endif
          GB_CTYPE *restrict Cx = (GB_CTYPE *) C->x ;

    //--------------------------------------------------------------------------
    // C += A*B, no mask, A bitmap/full, B sparse/hyper
    //--------------------------------------------------------------------------

    if (A_is_bitmap)
    { 
        // A is bitmap, B is sparse/hyper
        #undef  GB_A_IS_BITMAP
        #define GB_A_IS_BITMAP 1
        #include "GB_AxB_saxpy5_template.c"
    }
    else
    { 
        // A is full, B is sparse/hyper
        #undef  GB_A_IS_BITMAP
        #define GB_A_IS_BITMAP 0
        #include "GB_AxB_saxpy5_template.c"
    }
}

#undef GB_A_IS_BITMAP
#undef GB_B_IS_HYPER

