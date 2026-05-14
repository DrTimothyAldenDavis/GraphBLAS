//------------------------------------------------------------------------------
// gbreduce: reduce a sparse matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbreduce is an interface to GrB_Matrix_reduce_Monoid_Scalar.

// Usage:

//  cout = gbreduce (op, A)
//  cout = gbreduce (op, A, desc)
//  cout = gbreduce (cin, accum, op, A, desc)

// If cin is not present then it is implicitly a 1-by-1 matrix with no entries.

#define FREE_WORK                   \
    GrB_Matrix_free (&C_shallow) ;  \
    GrB_Matrix_free (&A_shallow) ;  \
    GrB_Descriptor_free (&desc) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = GrB.reduce (cin, accum, op, A, desc)"

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
    GrB_Matrix *C_opaque = NULL, C = NULL, A = NULL,
        C_shallow = NULL, A_shallow = NULL ;
    GrB_Descriptor desc = NULL ;

    gbmx_usage (nargin >= 2 && nargin <= 5 && nargout <= 2, USAGE) ;
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

    CHECK_ERROR (nmatrices < 1 || nmatrices > 2 || nstrings < 1 || ncells > 0,
        USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 1)
    { 
        OK (gb_get_matrix (&A, &A_shallow, &(Matrix [0]))) ;
    }
    else // if (nmatrices == 2)
    { 
        OK (gb_get_deep   (&C, &C_shallow, &(Matrix [0]))) ;
        OK (gb_get_matrix (&A, &A_shallow, &(Matrix [1]))) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    GrB_Monoid monoid ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_monoid (&monoid, String [0], atype)) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype)) ;
        OK (gb_string_to_monoid (&monoid, String [1], atype)) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        // use the ztype of the monoid as the type of C
        OK (gb_monoid_type (&ctype, monoid)) ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (1, 1, A, NULL, &(gbdesc.fmt))) ;
        OK (gb_get_sparsity (A, NULL, &(gbdesc.sparsity))) ;
        OK (gb_new (&C, ctype, 1, 1, gbdesc.fmt, gbdesc.sparsity)) ;
    }

    //--------------------------------------------------------------------------
    // ensure C is 1-by-1
    //--------------------------------------------------------------------------

    uint64_t cnrows, cncols ;
    OK (GrB_Matrix_nrows (&cnrows, C)) ;
    OK (GrB_Matrix_ncols (&cncols, C)) ;
    if (cnrows != 1 || cncols != 1)
    { 
        ERROR ("cin must be a scalar", GrB_DIMENSION_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // compute C += reduce(A)
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_reduce_Monoid_Scalar ((GrB_Scalar) C, accum, monoid, A,
        desc)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, gbdesc.kind)) ;
    (*kind_output) = (double) gbdesc.kind ;
    gb_wrapup ( ) ;
}

