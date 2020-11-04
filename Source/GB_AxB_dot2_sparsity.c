//------------------------------------------------------------------------------
// GB_AxB_dot2_sparsity: determine sparsity structure for C, for C=A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mxm.h"

int GB_AxB_dot2_sparsity            // sparsity of C for C=A'*B or C<!M>=A'*B
(
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B              // input matrix
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A for dot2 A'*B sparsity", GB0) ;
    ASSERT_MATRIX_OK (B, "B for dot2 A'*B sparsity", GB0) ;

    //--------------------------------------------------------------------------
    // determine the sparsity structure of C
    //--------------------------------------------------------------------------

    double C_bitmap_size = ((double) A->vdim) * ((double) B->vdim) ;

    double anvec = GB_IS_HYPERSPARSE (A) ? A->nvec : A->vdim ;
    double bnvec = GB_IS_HYPERSPARSE (B) ? B->nvec : B->vdim ;
    double C_max_density = (anvec * bnvec) / GB_IMAX (C_bitmap_size, 1) ;

    double A_size = (double) GB_NNZ_HELD (A) ;
    double B_size = (double) GB_NNZ_HELD (B) ;
    int C_sparsity ;
    if ((C_bitmap_size < 8 * (A_size + B_size)) &&      // C is small
        (C_max_density > GB_BITMAP_SWITCH_DEFAULT))     // and likely dense
    { 
        // C is not too large and likely dense: use a bitmap.  GB_AxB_dot2
        // will be very efficient in this case.
        C_sparsity = GxB_BITMAP ;
    }
    else
    { 
        // C is very large or very sparse: construct it as sparse/hypersparse.
        // GB_AxB_dot2 might be very inefficient in this case, unless either
        // A or B are dense.  This choice is determined by GB_AxB_meta.
        C_sparsity = GB_IS_HYPERSPARSE (B) ? GxB_HYPERSPARSE : GxB_SPARSE ;
    }

    return (C_sparsity) ;
}

