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
    { 
        // matrix is already nonfull (hypersparse, sparse, or bitmap);
        // nothing to do
        return (GrB_SUCCESS) ;
    }
    if (A->sparsity & GxB_BITMAP)
    { 
        // C can become bitmap
        return (GB_convert_full_to_bitmap (A, Context)) ;
    }
    else if (A->sparsity & GxB_SPARSE)
    {
        // C can become sparse
        return (GB_convert_full_to_sparse (A, Context)) ;
    }
    else if (A->sparsity & GxB_HYPERSPARSE)
    { 
        // C can become hypersparse
        return (GB_convert_any_to_hyper (A, Context)) ;
    }
    else
    { 
        // none of the above conditions hold so make A bitmap
        return (GB_convert_full_to_bitmap (A, Context)) ;
    }
}

