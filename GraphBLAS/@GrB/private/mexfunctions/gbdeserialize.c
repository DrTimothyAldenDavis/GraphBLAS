//------------------------------------------------------------------------------
// gbdeserialize: deserialize a blob into a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbdeserialize is an interface to GrB_Matrix_deserialize.

// Usage:

// C = gbdeserialize (blob)

// The blob can be either a MATLAB or a @GrB matrix.  In either case, it must
// be dense (not sparse) with all entries present, and of type GrB_UINT8.
// C is returned as a @GrB matrix.

#define FREE_WORK                       \
    GrB_Matrix_free (&Blob_shallow) ;

#define FREE_ALL                        \
    FREE_WORK ;                         \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = GrB.deserialize (blob)"

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

    GrB_Matrix *C_opaque = NULL, C = NULL, Blob = NULL, Blob_shallow = NULL ;

    gbmx_usage ((nargin >= 1 || nargin <= 3) && nargout <= 1, USAGE) ;
    pargout [0] = gbmx_export_struct (&C_opaque) ;

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    CHECK_ERROR (Matrix [0].type != GrB_UINT8,
        "blob must be a uint8 dense matrix/vector") ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the blob, normally a row or column vector, but can be a dense matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&Blob, &Blob_shallow, &(Matrix [0]))) ;

    bool Blob_is_dense = false ;
    OK (gb_is_dense (&Blob_is_dense, Blob)) ;
    CHECK_ERROR (!Blob_is_dense, "blob must be a uint8 dense matrix/vector") ;

    uint64_t nvals ;
    OK (GrB_Matrix_nvals (&nvals, Blob)) ;
    uint64_t blob_memsize = nvals * sizeof (uint8_t) ;
    const void *blob = Blob->x ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_deserialize (&C, NULL, blob, blob_memsize)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, KIND_GRB)) ;
    gb_wrapup ( ) ;
}

