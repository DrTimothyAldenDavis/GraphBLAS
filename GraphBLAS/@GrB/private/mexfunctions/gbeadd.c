//------------------------------------------------------------------------------
// gbeadd: sparse matrix addition
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbeadd is an interface to GrB_Matrix_eWiseAdd_BinaryOp.
// Note that gbeadd and gbemult are nearly identical mexFunctions.

// Usage:

// C = gbeadd (binop, A, B)
// C = gbeadd (binop, A, B, desc)
// C = gbeadd (Cin, accum, binop, A, B, desc)
// C = gbeadd (Cin, M, binop, A, B, desc)
// C = gbeadd (Cin, M, accum, binop, A, B, desc)

// TODO:
// gbeadd (C, accum, binop, A, B, desc)
// gbeadd (C, M, binop, A, B, desc)
// gbeadd (C, M, accum, binop, A, B, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A, B, and the descriptor).

#define FREE_WORK                   \
    GrB_Matrix_free (&C_shallow) ;  \
    GrB_Matrix_free (&M_shallow) ;  \
    GrB_Matrix_free (&A_shallow) ;  \
    GrB_Matrix_free (&B_shallow) ;  \
    GrB_Descriptor_free (&desc) ;

#define FREE_ALL                    \
    FREE_WORK                       \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = GrB.eadd (Cin, M, accum, binop, A, B, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Type atype, btype, ctype = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, A = NULL, B = NULL,
        C_shallow = NULL, M_shallow = NULL, A_shallow = NULL, B_shallow = NULL ;
    GrB_Descriptor desc = NULL ;

    gbmx_usage (nargin >= 3 && nargin <= 7 && nargout <= 2, USAGE) ;
    pargout [0] = gbmx_export_struct (&C_opaque) ;
    pargout [1] = mxCreateDoubleScalar (0) ;
    double *kind_output = (double *) mxGetData (pargout [1]) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    CHECK_ERROR (nmatrices < 2 || nmatrices > 4 || nstrings < 1 || ncells > 0,
        USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 2)
    { 
        OK (gb_get_matrix (&A, &A_shallow, &(Matrix [0]))) ;
        OK (gb_get_matrix (&B, &B_shallow, &(Matrix [1]))) ;
    }
    else if (nmatrices == 3)
    { 
        OK (gb_get_deep   (&C, &C_shallow, &(Matrix [0]))) ;
        OK (gb_get_matrix (&A, &A_shallow, &(Matrix [1]))) ;
        OK (gb_get_matrix (&B, &B_shallow, &(Matrix [2]))) ;
    }
    else // if (nmatrices == 4)
    { 
        OK (gb_get_deep   (&C, &C_shallow, &(Matrix [0]))) ;
        OK (gb_get_matrix (&M, &M_shallow, &(Matrix [1]))) ;
        OK (gb_get_matrix (&A, &A_shallow, &(Matrix [2]))) ;
        OK (gb_get_matrix (&B, &B_shallow, &(Matrix [3]))) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL, op = NULL ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_binop (&op, String [0], atype, btype)) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype)) ;
        OK (gb_string_to_binop (&op   , String [1], atype, btype)) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        // get the descriptor contents to determine if A is transposed
        int in0 ;
        OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;

        // get the size of A
        uint64_t anrows, ancols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        // determine the size of C
        uint64_t cnrows = (A_transpose) ? ancols : anrows ;
        uint64_t cncols = (A_transpose) ? anrows : ancols ;

        // use the ztype of the op as the type of C
        OK (gb_binaryop_ztype (&ctype, op)) ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, cncols, A, B, &(gbdesc.fmt))) ;
        OK (gb_get_sparsity (A, B, &(gbdesc.sparsity))) ;
        OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += A+B
    //--------------------------------------------------------------------------

    OK1 (C, GrB_Matrix_eWiseAdd_BinaryOp (C, M, accum, op, A, B, desc)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, gbdesc.kind)) ;
    (*kind_output) = (double) gbdesc.kind ;
    gb_wrapup ( ) ;
}

