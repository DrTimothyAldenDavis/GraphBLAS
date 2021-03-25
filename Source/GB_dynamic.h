//------------------------------------------------------------------------------
// GB_dynamic.h: convert a matrix to/from static/dynamic header
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_to_dynamic
(
    // output
    GrB_Matrix *Ahandle,        // output matrix A with dynamic header
    bool *A_input_is_static,    // if true, A_input has a static header
    // input
    GrB_Matrix A_input,         // input matrix with static or dynamic header
    GB_Context Context
) ;

void GB_to_static
(
    // output
    GrB_Matrix A_static,        // output matrix with static header
    // input
    GrB_Matrix *Ahandle,        // input matrix with dynamic header
    GB_Context Context
) ;

