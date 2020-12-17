//------------------------------------------------------------------------------
// GB_meta16_methods: methods for GB_meta16_factory.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    // declare macros that depend on the sparsity of A and B
    #include "GB_meta16_definitions.h"

    // include the method
    #if defined ( GB_DOT4 )
    #include "GB_AxB_dot4_template.c"
    #elif defined ( GB_DOT3_PHASE1 )
    #include "GB_AxB_dot3_phase1_template.c"
    #elif defined ( GB_DOT3_PHASE2 )
    #include "GB_AxB_dot3_template.c"
    #elif defined ( GB_DOT2 )
    #include "GB_AxB_dot2_template.c"
    #elif defined ( GB_SAXPY_COARSE_GUSTAVSON_PHASE1 )
    #include "GB_AxB_saxpy3_coarseGus_phase1.c"
    #elif defined ( GB_SAXPY_COARSE_GUSTAVSON_PHASE5 )
    #include "GB_AxB_saxpy3_coarseGus_phase5.c"
    #endif

    // undefine the macros that define the A and B sparsity
    #undef GB_A_IS_SPARSE
    #undef GB_A_IS_HYPER
    #undef GB_A_IS_BITMAP
    #undef GB_A_IS_FULL
    #undef GB_B_IS_SPARSE
    #undef GB_B_IS_HYPER
    #undef GB_B_IS_BITMAP
    #undef GB_B_IS_FULL
}

