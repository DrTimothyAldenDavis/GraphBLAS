//------------------------------------------------------------------------------
// GB_bitmap_AxB_saxpy: compute C=A*B, C<M>=A*B, or C<!M>=A*B; C bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_bitmap_AxB_saxpy.h"
#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GB_bitmap_AxB_saxpy: compute C=A*B, C<M>=A*B, or C<!M>=A*B
//------------------------------------------------------------------------------

GrB_Info GB_bitmap_AxB_saxpy        // C = A*B where C is bitmap or full
(
    GrB_Matrix *Chandle,            // output matrix (not computed in-place)
    const int C_sparsity,
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // mask always applied if present
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    (*mask_applied) = false ;
    ASSERT (Chandle != NULL) ;
    ASSERT (*Chandle == NULL) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for bitmap saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;

    ASSERT_MATRIX_OK (A, "A for bitmap saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT_MATRIX_OK (B, "B for bitmap saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for bitmap saxpy A*B", GB0) ;
    ASSERT (A->vdim == B->vlen) ;

    ASSERT (C_sparsity == GxB_BITMAP || C_sparsity == GxB_FULL) ;

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    GrB_Type ctype = semiring->add->op->ztype ;
    int64_t cnzmax ;
    bool ok = GB_Index_multiply ((GrB_Index *) &cnzmax, A->vlen, B->vdim) ;
    if (!ok)
    {
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_OK (GB_new_bix (Chandle, ctype, A->vlen, B->vdim, GB_Ap_null, true,
        C_sparsity, GB_HYPER_SWITCH_DEFAULT, -1, cnzmax, true, Context)) ;

    //--------------------------------------------------------------------------
    // scatter M into C
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------

    if (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B))
    {

        //-----------------------------------------------------
        // C                =               A     *     B
        //-----------------------------------------------------

        // bitmap           .               bitmap      hyper
        // bitmap           .               full        hyper
        // bitmap           .               bitmap      sparse
        // bitmap           .               full        sparse

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // bitmap           any             bitmap      hyper
        // bitmap           any             full        hyper
        // bitmap           any             bitmap      sparse
        // bitmap           any             full        sparse

        //-----------------------------------------------------
        // C               <!M>=            A     *     B
        //-----------------------------------------------------

        // bitmap           any             bitmap      hyper
        // bitmap           any             full        hyper
        // bitmap           any             bitmap      sparse
        // bitmap           any             full        sparse

        ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;

        // TODO
        ASSERT (0) ;

    }
    else if (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A))
    {

        //-----------------------------------------------------
        // C                =               A     *     B
        //-----------------------------------------------------

        // bitmap           .               hyper       bitmap
        // bitmap           .               sparse      bitmap
        // bitmap           .               hyper       full 
        // bitmap           .               sparse      full

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // bitmap           any             hyper       bitmap
        // bitmap           any             sparse      bitmap
        // bitmap           bitmap/full     hyper       full
        // bitmap           bitmap/full     sparse      full

        //-----------------------------------------------------
        // C               <!M>=            A     *     B
        //-----------------------------------------------------

        // bitmap           any             hyper       bitmap
        // bitmap           any             sparse      bitmap
        // bitmap           any             hyper       full 
        // bitmap           any             sparse      full

        ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;

        // TODO
        ASSERT (0) ;

    }
    else
    {

        //-----------------------------------------------------
        // C                =               A     *     B
        //-----------------------------------------------------

        // bitmap           .               bitmap      bitmap
        // bitmap           .               full        bitmap
        // bitmap           .               bitmap      full
        // full             .               full        full

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // bitmap           any             bitmap      bitmap
        // bitmap           any             full        bitmap
        // bitmap           bitmap/full     bitmap      full
        // bitmap           bitmap/full     full        full

        //-----------------------------------------------------
        // C               <!M>=            A     *     B
        //-----------------------------------------------------

        // bitmap           any             bitmap      bitmap
        // bitmap           any             full        bitmap
        // bitmap           any             bitmap      full
        // bitmap           any             full        full

        // TODO
        ASSERT (0) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*mask_applied) = (M != NULL) ;
    return (GrB_SUCCESS) ;
}

