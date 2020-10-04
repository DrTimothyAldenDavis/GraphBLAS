//------------------------------------------------------------------------------
// GB_AxB_saxpy: compute C=A*B, C<M>=A*B, or C<!M>=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_AxB_saxpy.h"
#include "GB_AxB_saxpy3.h"
#include "GB_AxB_bitmap_saxpy.h"

//------------------------------------------------------------------------------
// GB_AxB_saxpy: compute C=A*B, C<M>=A*B, or C<!M>=A*B
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy               // C = A*B using Gustavson/Hash/Bitmap
(
    GrB_Matrix *Chandle,            // output matrix (if not done in-place)
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, then mask was applied
    const GrB_Desc_Value AxB_method,
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

    ASSERT_MATRIX_OK_OR_NULL (M, "M for saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;

    ASSERT_MATRIX_OK (A, "A for saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT_MATRIX_OK (B, "B for saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for saxpy A*B", GB0) ;
    ASSERT (A->vdim == B->vlen) ;

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    int C_sparsity = GB_AxB_saxpy_sparsity (M, Mask_comp, A, B) ;

// HACK: saxpy does not yet construct C as bitmap (FIXME)
    if (C_sparsity == GxB_BITMAP || C_sparsity == GxB_FULL)
    {
        C_sparsity = GxB_SPARSE ;
    }

    //--------------------------------------------------------------------------
    // select the method to use
    //--------------------------------------------------------------------------

    // if (C_sparsity == GxB_HYPERSPARSE || C_sparsity == GxB_SPARSE)
    {

        //----------------------------------------------------------------------
        // C=A*B, C<M>=A*B or C<!M>=A*B: sparse Gustavson/Hash method
        //----------------------------------------------------------------------

        // GB_AxB_saxpy3 assumes C and B have the same sparsity structure
        C_sparsity = GB_IS_HYPERSPARSE (B) ? GxB_HYPERSPARSE : GxB_SPARSE ;
        return (GB_AxB_saxpy3 (Chandle, C_sparsity, M, Mask_comp, Mask_struct,
            A, B, semiring, flipxy, mask_applied, AxB_method, Context)) ;

    }
    #if 0
    else
    {

        //----------------------------------------------------------------------
        // C=A*B, C<M>=A*B or C<!M>=A*B: bitmap/full, possibly in-place 
        //----------------------------------------------------------------------

        return (GB_AxB_bitmap_saxpy (Chandle, C_sparsity, M, Mask_comp,
            Mask_struct, A, B, semiring, flipxy, mask_applied, Context)) ;
    }
    #endif
}

