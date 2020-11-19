//------------------------------------------------------------------------------
// GB_AxB_dot2_meta: C=A'*B, C<M>=A'*B or C<!M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// TODO: rename to GB_bitmap_AxB_dot_meta.c

//------------------------------------------------------------------------------

#include "GB_unused.h"
#include "GB_AxB_dot_cij.h"

// GB_DOT_ALWAYS_SAVE_CIJ: C(i,j) = cij
#define GB_DOT_ALWAYS_SAVE_CIJ      \
{                                   \
    int64_t pC = pC_start + i ;     \
    GB_PUTC (cij, pC) ;             \
    Cb [pC] = 1 ;                   \
    cnvals++ ;                      \
}                                   \

// GB_DOT_SAVE_CIJ: C(i,j) = cij, unless already done by GB_DOT
#if GB_IS_ANY_MONOID

    // for the ANY monoid, GB_DOT saves C(i,j) as soon as a value is found
    #define GB_DOT_SAVE_CIJ

#else

    // all other monoids: C(i,j) = cij if it exists
    #define GB_DOT_SAVE_CIJ             \
    {                                   \
        if (GB_CIJ_EXISTS)              \
        {                               \
            GB_DOT_ALWAYS_SAVE_CIJ ;    \
        }                               \
    }

#endif

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    int64_t cnvals = 0 ;

    ASSERT (GB_IS_BITMAP (C)) ;
    int8_t   *GB_RESTRICT Cb = C->b ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) (B_is_pattern ? NULL : B->x) ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) (A_is_pattern ? NULL : A->x) ;
    const int64_t cvlen = C->vlen ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int8_t  *GB_RESTRICT Bb = B->b ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const int64_t bnvec = B->nvec ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_sparse_or_hyper = B_is_sparse || B_is_hyper ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int8_t  *GB_RESTRICT Ab = A->b ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_sparse_or_hyper = A_is_sparse || A_is_hyper ;

    const int64_t vlen = A->vlen ;
    ASSERT (A->vlen == B->vlen) ;

    const int ntasks = naslice * nbslice ;

    //--------------------------------------------------------------------------
    // C=A'*B, C<M>=A'*B, or C<!M>=A'*B via dot products
    //--------------------------------------------------------------------------

    if (M == NULL)
    { 

        //----------------------------------------------------------------------
        // C = A'*B via dot products
        //----------------------------------------------------------------------

        #undef GB_MASK_IS_PRESENT
        #include "GB_AxB_dot2_meta2.c"

    }
    else
    {

        // TODO: if M is sparse, scatter into the C bitmap instead
        const int64_t *GB_RESTRICT Mp = M->p ;
        const int64_t *GB_RESTRICT Mh = M->h ;
        const int8_t  *GB_RESTRICT Mb = M->b ;
        const int64_t *GB_RESTRICT Mi = M->i ;
        const GB_void *GB_RESTRICT Mx = (GB_void *)
            (Mask_struct ? NULL : (M->x)) ;
        const size_t msize = M->type->size ;
        const int64_t mnvec = M->nvec ;
        const int64_t mvlen = M->vlen ;
        const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
        const bool M_is_sparse = GB_IS_SPARSE (M) ;
        const bool M_is_bitmap = GB_IS_BITMAP (M) ;
        const bool M_is_full = GB_IS_FULL (M) ;
        const bool M_is_bitmap_or_full = M_is_full || M_is_bitmap ;

        #if ( GB_IS_ANY_MONOID )
        if (B_is_bitmap && A_is_sparse && M_is_bitmap && Mask_struct
            && Mask_comp)
        {

            //------------------------------------------------------------------
            // C<#M,struct> = A'*B, special case
            //------------------------------------------------------------------

            // GB_ANY_SPECIALIZED is defined if the following conditions:
            // semirings: all built-in semirings with the ANY monoid
            // A: sparse
            // B: bitmap
            // M: bitmap
            // Mask_comp: true
            // Mask_struct: true

            GBURBLE ("(specialized) ") ;
            #define GB_ANY_SPECIALIZED
            #define GB_MASK_IS_PRESENT
            #define GB_A_IS_SPARSE_OR_HYPER 1
            #define GB_A_IS_BITMAP          0
            #define GB_A_IS_FULL            0
            #define GB_B_IS_SPARSE_OR_HYPER 0
            #define GB_B_IS_BITMAP          1
            #define GB_B_IS_FULL            0
            #include "GB_AxB_dot2_template.c"
            #undef  GB_ANY_SPECIALIZED
            #undef GB_MASK_IS_PRESENT

        }
        else
        #endif
        { 

            //------------------------------------------------------------------
            // C<M>=A'*B or C<!M>=A'*B via dot products
            //------------------------------------------------------------------

            #define GB_MASK_IS_PRESENT
            #include "GB_AxB_dot2_meta2.c"
            #undef GB_MASK_IS_PRESENT

        }
    }

    C->nvals = cnvals ;
}

