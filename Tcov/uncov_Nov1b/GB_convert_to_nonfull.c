//------------------------------------------------------------------------------
// GB_convert_to_nonfull: ensure a matrix is not full (hyper, sparse, or bitmap)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The matrix A must be converted from full to any other sparsity structure.
// The full sparsity structure cannot tolerate the deletion of any entry but
// the other three can.

#include "GB.h"

GrB_Info GB_convert_to_nonfull      // ensure a matrix is not full
(
    GrB_Matrix A,
    GB_Context Context
)
{

    if (!GB_IS_FULL (A))
    {   GB_cov[3090]++ ;
// NOT COVERED (3090):
GB_GOTCHA ;
        // matrix is already nonfull (hypersparse, sparse, or bitmap);
        // nothing to do
        return (GrB_SUCCESS) ;
    }
    else if (A->sparsity & GxB_BITMAP)
    {   GB_cov[3091]++ ;
// covered (3091): 2
        // C can become bitmap
        return (GB_convert_full_to_bitmap (A, Context)) ;
    }
    else if (A->sparsity & GxB_SPARSE
        || (A->vdim <= 1 & (A->sparsity & GxB_HYPERSPARSE)))
    {   GB_cov[3092]++ ;
// NOT COVERED (3092):
GB_GOTCHA ;
        // C can become sparse
        return (GB_convert_full_to_sparse (A, Context)) ;
    }
    else if (A->vdim > 1 && A->sparsity & GxB_HYPERSPARSE)
    {   GB_cov[3093]++ ;
// NOT COVERED (3093):
GB_GOTCHA ;
        // C can become hypersparse
        return (GB_convert_any_to_hyper (A, Context)) ;
    }
    else
    {   GB_cov[3094]++ ;
// NOT COVERED (3094):
GB_GOTCHA ;
        // none of the above conditions hold so make A bitmap
        return (GB_convert_full_to_bitmap (A, Context)) ;
    }
}

