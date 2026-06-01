//------------------------------------------------------------------------------
// gbfull: add identity values to a matrix so all entries are present
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard built-in
// sparse or full matrix.  The output is a GraphBLAS matrix by default, with
// all entries present, of the given type.  Entries are filled in with the id
// value, whose default value is zero.

// If desc.kind = 'grb', or if the descriptor is not present, the output is a
// GraphBLAS full matrix.  Otherwise the output is a built-in full matrix
// (desc.kind = 'full').   The two other cases, desc.kind = 'sparse' and
// 'builtin' are treated as 'full'.

// Usage:
//  C = gbfull (A)
//  C = gbfull (A, type)
//  C = gbfull (A, type, id)
//  C = gbfull (A, type, id, desc)

#define FREE_WORK                   \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Matrix_free (&id_to_free) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = gbfull (A, type, id, desc)"

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

    GrB_Matrix *C_opaque = NULL, C = NULL, A = NULL, A_to_free = NULL,
        id = NULL, id_to_free = NULL ;

    GBMX_USAGE (nargin >= 1 && nargin <= 4 && nargout <= 2, USAGE) ;

    pargout [0] = gbmx_export_struct (&C_opaque) ;
    pargout [1] = mxCreateDoubleScalar (0) ;
    double *kind_output = (double *) mxGetData (pargout [1]) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [2] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    struct gb_descriptor_struct gbdesc ;
    if (gbmx_mxarray_to_descriptor (&gbdesc, pargin [nargin-1]))
    { 
        // descriptor is present, remove it from further consideration
        nargin-- ;
    }

    char type_string [LEN+2] ;
    if (nargin > 1)
    { 
        gbmx_mxstring_to_string (type_string, LEN, pargin [1], "type") ;
    }

    if (nargin > 2)
    { 
        gbmx_get_matrix (&(Matrix [1]), pargin [2]) ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), err)) ;
    uint64_t nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // get the type of C
    //--------------------------------------------------------------------------

    GrB_Type type ;
    if (nargin > 1)
    { 
        type = gb_string_to_type (type_string) ;
    }
    else
    { 
        // the output type defaults to the same as the input type
        OK (GxB_Matrix_type (&type, A)) ;
    }

    //--------------------------------------------------------------------------
    // get the identity scalar
    //--------------------------------------------------------------------------

    if (nargin > 2)
    { 
        OK (gb_get_matrix (&id, &id_to_free, &(Matrix [1]), err)) ;
    }

    //--------------------------------------------------------------------------
    // finalize the kind and format
    //--------------------------------------------------------------------------

    // ignore gbdesc.kind = 'sparse' or 'builtin' and just use 'full' instead
    if (gbdesc.kind == KIND_SPARSE || gbdesc.kind == KIND_BUILTIN)
    { 
        gbdesc.kind = KIND_FULL ;
    }

    if (gbdesc.kind == KIND_FULL)
    { 
        // built-in matrices are always held by column
        gbdesc.fmt = GxB_BY_COL ;
    }
    else
    { 
        // A determines the format of C, unless defined by the descriptor
        OK (gb_get_format (nrows, ncols, A, NULL, &(gbdesc.fmt), err)) ;
    }

    //--------------------------------------------------------------------------
    // expand A to a full matrix
    //--------------------------------------------------------------------------

    OK (gb_expand_to_full (&C, A, type, gbdesc.fmt, id, err)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, gbdesc.kind, err)) ;
    (*kind_output) = (double) gbdesc.kind ;
    gb_wrapup ( ) ;
}

