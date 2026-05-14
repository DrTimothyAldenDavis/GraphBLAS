//------------------------------------------------------------------------------
// gb_is_column_vector: determine if A is a column vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_UTIL
#include "gb_interface.h"

GrB_Info gb_is_column_vector    // determine if A is a column vector
(
    // output:
    bool *is_column_vector,
    // input:
    GrB_Matrix A                // GrB_matrix to query
)
{ 

    if (A == NULL)
    { 
        (*is_column_vector) = false ;
        return (GrB_SUCCESS) ;
    }

    uint64_t ncols ;
    int sparsity, orientation ;

    OK (GrB_Matrix_get_INT32 (A, &sparsity, GxB_SPARSITY_STATUS)) ;
    OK (GrB_Matrix_get_INT32 (A, &orientation, GrB_STORAGE_ORIENTATION_HINT)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    (*is_column_vector) = (sparsity != GxB_HYPERSPARSE &&
        orientation == GrB_COLMAJOR && ncols == 1) ;

    return (GrB_SUCCESS) ;
}

