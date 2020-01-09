//------------------------------------------------------------------------------
// GB_dense.h: defintions for dense matrix operations 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_DENSE_H
#define GB_DENSE_H

#include "GB_ek_slice.h"

GrB_Info GB_dense_accum_sparse      // C += A where C is dense and A is sparse 
(
    GrB_Matrix C,                   // input/output matrix
    const GrB_Matrix A,             // input matrix
    const GrB_BinaryOp accum,       // operator to apply
    GB_Context Context
) ;

GrB_Info GB_dense_accum_scalar      // C += x where C is dense and x is a scalar 
(
    GrB_Matrix C,                   // input/output matrix
    const GB_void *scalar,          // input scalar
    const GrB_Type atype,           // type of the input scalar
    const GrB_BinaryOp accum,       // operator to apply
    GB_Context Context
) ;

GrB_Info GB_dense_expand_scalar     // C(:,:) = x; C is a matrix and x a scalar
(
    GrB_Matrix C,                   // input/output matrix
    const GB_void *scalar,          // input scalar
    const GrB_Type atype,           // type of the input scalar
    GB_Context Context
) ;

#endif

