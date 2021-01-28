//------------------------------------------------------------------------------
// GB_cuda_stringify_sparsity: determine the sparsity status of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_cuda_stringify.h"

//------------------------------------------------------------------------------
// GB_stringify_sparsity: define macros for sparsity structure
//------------------------------------------------------------------------------

void GB_stringify_sparsity  // construct macros for sparsity structure
(
    // output:
    char *sparsity_macros,  // macros that define the sparsity structure
    // intput:
    char *matrix_name,      // "C", "M", "A", or "B"
    GrB_Matrix A
)
{

    int ecode ;
    GB_enumify_sparsity (&ecode, A) ;
    GB_macrofy_sparsity (sparsity_macros, matrix_name, ecode) ;
}

//------------------------------------------------------------------------------
// GB_enumify_sparsity: enumerate the sparsity structure of a matrix
//------------------------------------------------------------------------------

void GB_enumify_sparsity    // enumerate the sparsity structure of a matrix
(
    // output:
    int *ecode,             // enumerated sparsity structure
    // input:
    GrB_Matrix A
)
{

    if (A == NULL)
    {
        // if A is NULL, pretend it is sparse
        e = 0 ; // (GxB_SPARSE) ;
    }
    else if (GB_IS_HYPERSPARSE (A))
    { 
        e = 1 ; // (GxB_HYPERSPARSE) ;
    }
    else if (GB_IS_FULL (A))
    { 
        e = 3 ; // (GxB_FULL) ;
    }
    else if (GB_IS_BITMAP (A))
    { 
        e = 2 ; // (GxB_BITMAP) ;
    }
    else
    { 
        e = 0 ; // (GxB_SPARSE) ;
    }
    (*ecode) = e ;
}

//------------------------------------------------------------------------------
// GB_macrofy_sparsity: define a macro for the sparsity structure of a matrix
//------------------------------------------------------------------------------

void GB_macrofy_sparsity    // construct macros for sparsity structure
(
    // output:
    char *sparsity_macros,  // macros that define the sparsity structure
    // input:
    char *matrix_name,      // "C", "M", "A", or "B"
    int ecode
)
{

    switch (ecode)
    {

        case 0 :
            snprintf (sparsity_macros, GB_CUDA_STRLEN,
                "#define %s_IS_SPARSE 1\n"
                "#define %s_IS_HYPER  0\n"
                "#define %s_IS_BITMAP 0\n"
                "#define %s_IS_FULL   0\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            break ;

        case 1 :
            snprintf (sparsity_macros, GB_CUDA_STRLEN,
                "#define %s_IS_SPARSE 0\n"
                "#define %s_IS_HYPER  1\n"
                "#define %s_IS_BITMAP 0\n"
                "#define %s_IS_FULL   0\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            break ;

        case 2 :
            snprintf (sparsity_macros, GB_CUDA_STRLEN,
                "#define %s_IS_SPARSE 0\n"
                "#define %s_IS_HYPER  0\n"
                "#define %s_IS_BITMAP 1\n"
                "#define %s_IS_FULL   0\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            break ;

        case 3 :
            snprintf (sparsity_macros, GB_CUDA_STRLEN,
                "#define %s_IS_SPARSE 0\n"
                "#define %s_IS_HYPER  0\n"
                "#define %s_IS_BITMAP 0\n"
                "#define %s_IS_FULL   1\n",
                matrix_name, matrix_name, matrix_name, matrix_name) ;
            break ;

        default :
    }
}

