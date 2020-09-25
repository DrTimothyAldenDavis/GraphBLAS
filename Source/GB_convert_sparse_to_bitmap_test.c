//------------------------------------------------------------------------------
// GB_convert_sparse_to_bitmap_test: test conversion of hyper/sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Returns true if a sparse or hypersparse matrix should be converted to bitmap
// (or full if all entries are present, but in that case, this function is not
// called).  Returns false if the matrix should stay hypersparse/sparse.

// If A is m-by-n and A->sparsity is GxB_AUTO_SPARSITY with the default
// A->bitmap_switch: the matrix switches to bitmap if nnz(A)/(m*n) > (1/8).
// That, if the density is 12.5% or more.  A bitmap matrix switches to sparse
// if nnz(A)/(m*n) <= (1/512), that is, if the matrix density is about 0.2% or
// less.  A matrix whose density is between 0.2% and 12.5% remains in its
// current state.

// A->bitmap_switch is normally a fraction in range 0 to 1, and is (1/8) by
// default.  If set to 1 or more, A never becomes bitmap.

// These default rules may change in future releases of SuiteSparse:GraphBLAS.

// If A->sparsity is GxB_HYPERSPARSE, GxB_SPARSE, GxB_BITMAP, or GxB_FULL,
// then this automatic test is not used.

// This rule is not used if all entries are present.  In that case, the matrix
// becomes full, not bitmap.

#include "GB.h"

bool GB_convert_sparse_to_bitmap_test    // test for hyper/sparse to bitmap
(
    float bitmap_switch,    // A->bitmap_switch
    int64_t anz,            // # of entries in A = GB_NNZ (A)
    int64_t vlen,           // A->vlen
    int64_t vdim            // A->vdim
)
{
    return (false) ; // TODO:BITMAP for now, do not convert to bitmap

#if 0
    // current number of entries in the matrix or vector
    float nnz = (float) anz ;

    // maximum number of entries in the matrix or vector
    float nnz_dense = ((float) vlen) * ((float) vdim) ;

    // A should switch to bitmap if the following condition is true:
    return (nnz > bitmap_switch * nnz_dense &&
            nnz_dense < (float) GxB_INDEX_MAX) ;
#endif
}

