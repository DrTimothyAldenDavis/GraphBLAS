//------------------------------------------------------------------------------
// GB_AxB_saxpy_sparsity: determine the sparsity structure for C<M or !M>=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Determines the sparsity structure for C, for computing C=A*B, C<M>=A*B, or
// C<!M>=A*B, based on the sparsity structures of C (on input), M, A, and B,
// and whether or not M is complemented.

// TODO: write the following AxB_saxpy kernels:  the bitmap/full cases are
// treated the same, for now, and the sparse/hypersparse cases are also handled
// the same.

// TODO: When A or B are bitmapped or full, they can be transposed in-place.

//------------------------------------------------------------------------------
// C       <no mask, M, !M>       +=       A   *   B
//------------------------------------------------------------------------------

// sparse  any                             any     any:

//      GB_AxB_saxpy3 (rename to GB_sparse_AxB_saxpy):
//      takes inputs of any type but always produces a sparse or hypersparse
//      matrix C.  It cannot modify C in place.

//------------------------------------------------------------------------------

// bitmap  sparse M or !M                   bitmap   sparse

//      GB_bitmap_AxB_saxpy_M_sparse_A_bitmap_B_sparse:
//      A is bitmap or full.  B is sparse or hypersparse.
//      scatter M or !M into the C bitmap.  A and B can be sliced the same way
//      as dot2.  No atomics.  C can be modified in-place if the accum operator
//      matches the semiring monoid, with no change to the code, except that
//      when not in-place, C must be calloc'd first.

// bitmap  none                             bitmap   sparse

//      GB_bitmap_AxB_saxpy_M_none_A_bitmap_B_sparse:
//      same as GB_bitmap_AxB_saxpy_M_sparse_A_bitmap_B_sparse, just no mask.

// bitmap  bitmap or full, M or !M          bitmap   sparse

//      GB_bitmap_AxB_saxpy_M_bitmap_A_bitmap_B_sparse:
//      A is bitmap or full.  B is sparse or hypersparse.
//      same as GB_bitmap_AxB_saxpy_M_sparse_A_bitmap_B_sparse, except that
//      M can be used in-place, instead of being copied into the C bitmap.

//------------------------------------------------------------------------------

// bitmap  none                             sparse  bitmap

//      GB_bitmap_AxB_saxpy_M_none_A_sparse_B_bitmap:
//      Like GB_AxB_saxpy_C_sparse, except that all tasks are coarse/fine
//      Gustavson (fine Gustavson with atomics).  No symbolic pre-analysis,
//      and no Gustavson workspace.

// bitmap  sparse M or !M                   sparse  bitmap

//      GB_bitmap_AxB_saxpy_M_sparse_A_sparse_B_bitmap:
//      Like GB_bitmap_AxB_saxpy_M_none_A_sparse_B_bitmap.  Scatter M into
//      the C bitmap.

// bitmap  bitmap or full, M or !M          sparse   bitmap

//      GB_bitmap_AxB_saxpy_M_bitmap_A_sparse_B_bitmap:
//      Like GB_bitmap_AxB_saxpy_M_sparse_A_sparse_B_bitmap, except use M
//      in-place; do not scatter into the C bitmap.

//------------------------------------------------------------------------------

// bitmap  none                             bitmap  bitmap

//      GB_bitmap_AxB_saxpy_M_none_A_bitmap_B_bitmap:
//      This method can used arbitrary tiling methods.  Divide up the C matrix
//      into K-by-K tiles for some chosen constant K, and compute each C(i,j)
//      tile independently.  Pick the tilesize based on L1 cache-size: probably
//      K=32 or K=64.  If # tiles < # threads, could use more than one thread
//      per C(i,j) tile, and accumulate when done.

// bitmap  sparse M or !M                   bitmap  bitmap

//      GB_bitmap_AxB_saxpy_M_sparse_A_bitmap_B_bitmap:
//      Like GB_bitmap_AxB_saxpy_M_none_A_bitmap_B_bitmap, except scatter
//      M and !M into the C bitmap.  Before computing the C(i,j) tile, check
//      the mask to see if any entry is allowed to be modified by the mask,
//      and skip the work.  Then clear the mask from the C(i,j) tile.
//      If there are very few entries to compute in the C(i,j) tile, could
//      use a dot-product method instead, to compute each tile multiply,
//      C(i,j)+=A(i,k)*B(k,j).

// bitmap  bitmap or full, M or !M          bitmap   bitmap

//      GB_bitmap_AxB_saxpy_M_bitmap_A_bitmap_B_bitmap:
//      Like GB_bitmap_AxB_saxpy_M_sparse_A_bitmap_B_bitmap, except use M or
//      !M in-place.

//------------------------------------------------------------------------------

#include "GB_AxB_saxpy.h"

int GB_AxB_saxpy_sparsity           // return the sparsity structure for C
(
    // input:
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,             // input A matrix
    const GrB_Matrix B              // input B matrix
)
{

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    int C_sparsity ;

    double m = (double) A->vlen ;
    double k = (double) A->vdim ;
    double n = (double) B->vdim ;
    bool C_is_large = (m*n) > 4 * (m*k + k*n) ;

    int M_sparsity = (M == NULL) ? 0 : GB_sparsity (M) ;
    int B_sparsity = GB_sparsity (B) ;
    int A_sparsity = GB_sparsity (A) ;
    bool M_is_hyper  = (M_sparsity == GxB_HYPERSPARSE) ;
    bool M_is_sparse = (M_sparsity == GxB_SPARSE) ;

    if (M != NULL && !Mask_comp && (M_is_hyper || M_is_sparse))
    {

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // hyper            hyper           any         any  
        // hyper            sparse          any         hyper
        // sparse           sparse          any         sparse/bitmap/full

        // The non-empty columns of C are a subset of the non-empty columns of
        // B, so in general, if B is hypersparse, so is C.

        // When M is hypersparse or sparse, and not complemented, C has the
        // same format as M, except when B is hypersparse, in which case C is
        // also hypersparse.

        // BFS: will not come here since M is bitmap/full

        if (B_sparsity == GxB_HYPERSPARSE)
        {
            C_sparsity = GxB_HYPERSPARSE ;
        }
        else
        {
            C_sparsity = M_sparsity ;
        }

    }
    else
    {

        //-----------------------------------------------------
        // C                =               A     *     B
        //-----------------------------------------------------

        // hyper            .               hyper       hyper
        // hyper            .               sparse      hyper
        // hyper/bitmap     .               bitmap      hyper
        // hyper/bitmap     .               full        hyper

        // sparse           .               hyper       sparse
        // sparse           .               sparse      sparse
        // sparse/bitmap    .               bitmap      sparse
        // sparse/bitmap    .               full        sparse

        // sparse/bitmap    .               hyper       bitmap
        // sparse/bitmap    .               sparse      bitmap
        // bitmap           .               bitmap      bitmap
        // bitmap           .               full        bitmap

        // sparse/bitmap    .               hyper       full 
        // sparse/bitmap    .               sparse      full
        // bitmap           .               bitmap      full
        // full             .               full        full

        //-----------------------------------------------------
        // C               <M>=             A     *     B
        //-----------------------------------------------------

        // hyper            any             hyper       hyper
        // hyper            any             sparse      hyper
        // hyper/bitmap     any             bitmap      hyper
        // hyper/bitmap     any             full        hyper

        // sparse           any             hyper       sparse
        // sparse           any             sparse      sparse
        // sparse/bitmap    any             bitmap      sparse
        // sparse/bitmap    any             full        sparse

        // sparse/bitmap    any             hyper       bitmap
        // sparse/bitmap    any             sparse      bitmap
        // bitmap           any             bitmap      bitmap
        // bitmap           any             full        bitmap

        // sparse/bitmap    bitmap/full     hyper       full    (*)
        // sparse/bitmap    bitmap/full     sparse      full    (*)
        // bitmap           bitmap/full     bitmap      full    (*)
        // bitmap           bitmap/full     full        full    (*)

        // (*): if M hyper/sparse, then C is hyper/sparse; see above

        //-----------------------------------------------------
        // C               <!M>=            A     *     B
        //-----------------------------------------------------

        // hyper            any             hyper       hyper
        // hyper            any             sparse      hyper
        // hyper/bitmap     any             bitmap      hyper
        // hyper/bitmap     any             full        hyper

        // sparse           any             hyper       sparse
        // sparse           any             sparse      sparse
        // sparse/bitmap    any             bitmap      sparse
        // sparse/bitmap    any             full        sparse

        // sparse/bitmap    any             hyper       bitmap
        // sparse/bitmap    any             sparse      bitmap
        // bitmap           any             bitmap      bitmap
        // bitmap           any             full        bitmap

        // sparse/bitmap    any             hyper       full 
        // sparse/bitmap    any             sparse      full
        // bitmap           any             bitmap      full
        // bitmap           any             full        full

        // If M is complemented, or not complemented and bitmap/full, then C
        // has the same sparsity as listed above, except when A and B are both
        // full.

        // For the cases where C is labelled as hyper/bitmap or sparse/bitmap:
        // Let A by m-by-k, let B by k-by-n, then C is m-by-n.  If m*n is much
        // larger than (m*k + k*n), then always construct C as sparse/hyper,
        // not bitmap.

        // BFS: C<!M>=A*B, no accum, M is complemented, not structural, full
        // (or bitmap in the future).  So C cannotb e not computed in-place.
        // Also, C is aliased with B.

        switch (B_sparsity)
        {
            case GxB_HYPERSPARSE :
            case GxB_SPARSE :
                switch (A_sparsity)
                {
                    case GxB_HYPERSPARSE :
                    case GxB_SPARSE :
                        C_sparsity = B_sparsity ;
                        break ;
                    case GxB_BITMAP :
                    case GxB_FULL :
                        C_sparsity = C_is_large ? B_sparsity : GxB_BITMAP ;
                        break ;
                    default: ;
                }
                break ;
            case GxB_BITMAP :
                // BFS comes here: B is bitmap
            case GxB_FULL : switch (A_sparsity)
                {
                    case GxB_HYPERSPARSE :
                    case GxB_SPARSE :
                        // BFS comes here: A is sparse, and C is not huge
                        C_sparsity = C_is_large ? GxB_SPARSE : GxB_BITMAP ;
                        break ;
                    case GxB_BITMAP :
                        C_sparsity = GxB_BITMAP ;
                        break ;
                    case GxB_FULL :
                        C_sparsity = (M == NULL) ? GxB_FULL : GxB_BITMAP ;
                        break ;
                    default: ;
                }
                break ;
            default: ;
        }
    }

    return (C_sparsity) ;
}

