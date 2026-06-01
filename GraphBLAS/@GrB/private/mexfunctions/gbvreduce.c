//------------------------------------------------------------------------------
// gbvreduce: reduce a matrix to a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbvreduce is an interface to GrB_Matrix_reduce.

// Usage:

//  C = gbvreduce (op, A)
//  C = gbvreduce (op, A, desc)
//  C = gbvreduce (Cin, M, op, A, desc)
//  C = gbvreduce (Cin, accum, op, A, desc)
//  C = gbvreduce (Cin, M, accum, op, A, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A and the descriptor).

#define FREE_WORK                   \
    GrB_Matrix_free (&M_to_free) ;  \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Descriptor_free (&desc) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = GrB.vreduce (Cin, M, accum, op, A, desc)"

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

    GrB_Type atype, ctype = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, A = NULL,
        M_to_free = NULL, A_to_free = NULL ;
    GrB_Descriptor desc = NULL ;

    GBMX_USAGE (nargin >= 2 && nargin <= 6 && nargout <= 2, USAGE) ;

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

    CHECK_ERROR (nmatrices < 1 || nmatrices > 3 || nstrings < 1 || ncells > 0,
        USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor_mxm (&desc, &gbdesc, err)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 1)
    { 
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), err)) ;
    }
    else if (nmatrices == 2)
    { 
        OK (gb_get_deep   (&C,             &(Matrix [0]), err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [1]), err)) ;
    }
    else // if (nmatrices == 3)
    { 
        OK (gb_get_deep   (&C,             &(Matrix [0]), err)) ;
        OK (gb_get_matrix (&M, &M_to_free, &(Matrix [1]), err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [2]), err)) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        bool is_column_vector ;
        OK (gb_is_column_vector (&is_column_vector, C, err)) ;
        CHECK_ERROR (!is_column_vector, "Cin must be a column vector") ;
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    GrB_Monoid monoid ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_monoid (&monoid, String [0], atype, err)) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype, err)) ;
        OK (gb_string_to_monoid (&monoid, String [1], atype, err)) ;
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

        // determine the size of the vector C
        uint64_t cnrows = (A_transpose) ? ancols : anrows ;

        // use the ztype of the monoid as the type of C
        OK (gb_monoid_type (&ctype, monoid, err)) ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, 1, A, NULL, &(gbdesc.fmt), err)) ;
        OK (gb_get_sparsity (A, NULL, &(gbdesc.sparsity), err)) ;
        OK (gb_new (&C, ctype, cnrows, 1, gbdesc.fmt, gbdesc.sparsity, err)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += reduce(A)
    //--------------------------------------------------------------------------

    OK1 (C, GrB_Matrix_reduce_Monoid ((GrB_Vector) C, (GrB_Vector) M,
        accum, monoid, A, desc)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, gbdesc.kind, err)) ;
    (*kind_output) = (double) gbdesc.kind ;
    gb_wrapup ( ) ;
}

