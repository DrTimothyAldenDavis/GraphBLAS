//------------------------------------------------------------------------------
// GB_to_hyper_test: test if a matrix should convert to hypersparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Returns true if a sparse matrix should be converted to hypersparse.
// Returns false if the matrix should stay sparse.

// A matrix with vdim <= 1 must always be sparse, not hypersparse;
// that is, a GrB_Vector is never hypersparse.

#include "GB.h"

bool GB_to_hyper_test       // test for conversion from sparse to hypersparse
(
    double hyper_ratio,     // A->hyper_ratio
    int64_t k,              // # of non-empty vectors of A, an estimate is OK,
                            // but normally A->nvec_nonempty
    int64_t vdim            // A->vdim
)
{

    // get the vector dimension of this matrix
    float n = (float) vdim ;

    // get the hyper ratio
    float r = (float) hyper_ratio ;

    // ensure k is in the range 0 to n, inclusive
    k = GB_IMAX (k, 0) ;
    k = GB_IMIN (k, n) ;

    return (n > 1 && (((float) k) <= n * r)) ;
}

