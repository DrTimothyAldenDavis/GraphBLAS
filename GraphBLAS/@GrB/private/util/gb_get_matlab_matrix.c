//------------------------------------------------------------------------------
// gb_get_matlab_matrix: get a MATLAB matrix argument
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The A->[pix] content is tagged GxB_IS_READONLY, so the arena (MATLAB
// mxMalloc) doesn't matter.  However, the pointers are tagged as in the
// MXARENA anyway.  The header of A is placed in the arena determined by the
// input parameter.

#define GB_UTIL

#define FREE_WORK                       \
    GxB_Container_free (&Container) ;

#define FREE_ALL                        \
    FREE_WORK                           \
    GrB_Matrix_free (&A) ;

#include "gb_interface.h"

GrB_Info gb_get_matlab_matrix    // shallow copy of MATLAB sparse matrix
(
    // output
    GrB_Matrix *A_handle,   // content of A is tagged GxB_IS_READONLY
    // input
    gb_matrix matrix,       // contents of a MATLAB matrix
    const int arena,
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // load the content of the MATLAB matrix into the Container, as readonly
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL ;

    GxB_Container Container = NULL ;

    OK (GxB_Container_new_arena (&Container, arena, arena)) ;

    Container->nrows = matrix->nrows ;
    Container->ncols = matrix->ncols ;
    Container->nvals = matrix->nvals ;
    Container->nrows_nonempty = -1 ;
    Container->ncols_nonempty = -1 ;
    Container->orientation = GrB_COLMAJOR ;
    Container->iso = false ;
    Container->jumbled = false ;

    if (matrix->sparsity == GxB_SPARSE)
    { 
        // import the matrix in CSC format (all-64-bit)
        uint64_t Xp_memsize = (matrix->ncols + 1) * sizeof (uint64_t) ;
        uint64_t Xi_memsize = matrix->nvals * sizeof (uint64_t) ;
        OK (GxB_Vector_load (Container->p, &(matrix->p), GrB_UINT64,
            matrix->ncols + 1, Xp_memsize, GxB_IS_READONLY + MXARENA, NULL)) ;
        OK (GxB_Vector_load (Container->i,  &(matrix->i), GrB_UINT64,
            matrix->nvals, Xi_memsize, GxB_IS_READONLY + MXARENA, NULL)) ;
        Container->format = GxB_SPARSE ;
    }
    else
    { 
        // import a full matrix
        Container->format = GxB_FULL ;
    }

    uint64_t Xx_memsize = matrix->nvals * matrix->typesize  ;
    OK (GxB_Vector_load (Container->x, &(matrix->x), matrix->type,
        matrix->nvals, Xx_memsize, GxB_IS_READONLY + MXARENA, NULL)) ;

    //--------------------------------------------------------------------------
    // unload the Container into A
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_new_arena (&A, matrix->type, matrix->nrows, matrix->ncols,
        arena, arena)) ;
    OK (GxB_load_Matrix_from_Container (A, Container, NULL)) ;
    (*A_handle) = A ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    return (GrB_SUCCESS) ;
}

