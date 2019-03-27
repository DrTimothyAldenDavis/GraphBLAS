//------------------------------------------------------------------------------
// GB_AxB_dot_meta: C=A'*B or C<M>=A'*B via dot productes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A and B
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_1_OF_2)
    ;
    #else
    const GB_ATYPE *restrict Ax = A_is_pattern ? NULL : A->x ;
    const GB_BTYPE *restrict Bx = B_is_pattern ? NULL : B->x ;
    #endif

    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ai = A->i ;
    int64_t anvec = A->nvec ;
    bool A_is_hyper = GB_IS_HYPER (A) ;

    const int64_t *restrict Bi = B->i ;
    int64_t bvlen = B->vlen ;

    //--------------------------------------------------------------------------
    // start the construction of C
    //--------------------------------------------------------------------------

    #if defined ( GB_PHASE_1_OF_2)
    ;
    #elif defined ( GB_PHASE_2_OF_2)
    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ci = C->i ;
    #else
    int64_t *restrict Ci = C->i ;
    int64_t jlast, cnz, cnz_last ;
    GB_jstartup (C, &jlast, &cnz, &cnz_last) ;
    #endif

    //--------------------------------------------------------------------------
    // C=A'*B, C<M>=A'*B, or C<!M>=A'*B via dot products
    //--------------------------------------------------------------------------

    if (M == NULL)
    { 

        // C = A'*B via dot products
        #include "GB_AxB_dot_nomask.c"

    }
    else
    {

        //----------------------------------------------------------------------
        // get M
        //----------------------------------------------------------------------

        const int64_t *restrict Mp = M->p ;
        const int64_t *restrict Mh = M->h ;
        const int64_t *restrict Mi = M->i ;
        const GB_void *restrict Mx = M->x ;
        GB_cast_function cast_M = GB_cast_factory (GB_BOOL_code, M->type->code);
        size_t msize = M->type->size ;
        const int64_t mnvec = M->nvec ;
        int64_t mpleft = 0 ;
        int64_t mpright = mnvec - 1 ;
        bool M_is_hyper = GB_IS_HYPER (M) ;

        if (Mask_comp)
        {
            // C<!M> = A'*B via dot products
            #include "GB_AxB_dot_compmask.c"
        }
        else
        { 
            // C<M> = A'*B via dot products
            #include "GB_AxB_dot_mask.c"
        }
    }
}

