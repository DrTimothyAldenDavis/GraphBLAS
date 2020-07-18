//------------------------------------------------------------------------------
// GB_AxB_dot2_meta: C=A'*B or C<!M>=A'*B via dot productes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_2_OF_2)
    int64_t  *GB_RESTRICT Cp = C->p ;
    int64_t  *GB_RESTRICT Ci = C->i ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;
    const GB_BTYPE *GB_RESTRICT Bx = (GB_BTYPE *) (B_is_pattern ? NULL : B->x) ;
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) (A_is_pattern ? NULL : A->x) ;
    #endif

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    int64_t bnvec = B->nvec ;
    int64_t bvlen = B->vlen ;

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    int64_t anvec = A->nvec ;
    int64_t avlen = A->vlen ;

    //--------------------------------------------------------------------------
    // C=A'*B or C<!M>=A'*B via dot products
    //--------------------------------------------------------------------------

    if (M == NULL)
    { 

        // C = A'*B via dot products
        #include "GB_AxB_dot2_nomask.c"

    }
    else
    { 

        //----------------------------------------------------------------------
        // get M
        //----------------------------------------------------------------------

        const int64_t *GB_RESTRICT Mp = M->p ;
        const int64_t *GB_RESTRICT Mh = M->h ;
        const int64_t *GB_RESTRICT Mi = M->i ;
        const GB_void *GB_RESTRICT Mx ;
        Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;
        size_t msize = M->type->size ;
        const int64_t mnvec = M->nvec ;
        const int64_t mvlen = M->vlen ;
        bool M_is_hyper = GB_IS_HYPER (M) ;

        // C<!M> = A'*B via dot products
        #include "GB_AxB_dot2_compmask.c"
    }

}

