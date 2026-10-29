//------------------------------------------------------------------------------
// GB_ewise_fulln_template: C = A+B where all 3 matrices are full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// No matrix is iso.  This template is not used for any generic kernels, and
// the operator is not an index binary op.

#include "include/GB_ewise_shared_definitions.h"

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    // any matrix may be aliased to any other (C==A, C==B, and/or A==B)
    GB_A_TYPE *Ax = (GB_A_TYPE *) A->x ;
    GB_B_TYPE *Bx = (GB_B_TYPE *) B->x ;
    GB_C_TYPE *Cx = (GB_C_TYPE *) C->x ;
    GB_C_NVALS (cnz) ;          // const int64_t cnz = GB_nnz (C) ;
    ASSERT (GB_IS_FULL (A)) ;
    ASSERT (GB_IS_FULL (B)) ;
    ASSERT (GB_IS_FULL (C)) ;
    ASSERT (!C->iso) ;
    ASSERT (!A->iso) ;
    ASSERT (!B->iso) ;
    int64_t p ;

    //--------------------------------------------------------------------------
    // C = A+B where all 3 matrices are full
    //--------------------------------------------------------------------------

    // note that A, B and C may all be aliased to each other, in any way
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    { 
        // C (i,j) = A (i,j) + B (i,j)
        GB_EWISE_AIJ_OP_BIJ (Cx, p, Ax, p, false, Bx, p, false, 0, 0) ;
    }
}

