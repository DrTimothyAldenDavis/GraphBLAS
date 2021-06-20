//------------------------------------------------------------------------------
// GB_transpose_cast: transpose and typecast
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The transpose is not in-place.  No operator is applied.  C = (ctype) A' is
// computed, with typecasting if ctype is not equal to A->type.

#include "GB_transpose.h"

GrB_Info GB_transpose_cast      // C=(ctype)A' with typecast, not in-place
(
    GrB_Matrix C,               // output matrix C, not in place
    GrB_Type ctype,             // desired type of C; if NULL use A->type
    const bool C_is_csc,        // desired CSR/CSC format of C
    const GrB_Matrix A,         // input matrix; C != A
    GB_Context Context
)
{ 
    ASSERT (C != A && !GB_aliased (C, A)) ;

    // C = (ctype) A', or C = A' if ctype is NULL
    return (GB_transpose (C, ctype, C_is_csc, A, NULL, NULL, NULL, false,
        Context)) ;
}

