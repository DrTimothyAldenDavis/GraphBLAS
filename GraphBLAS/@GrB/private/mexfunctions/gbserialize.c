//------------------------------------------------------------------------------
// gbserialize: serialize a matrix into a blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// gbserialize is an interface to GxB_Matrix_serialize.

// Usage:

// blob = gbserialize (A)

#include "gb_interface.h"

#define USAGE "usage: blob = GrB.serialize (A)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 1 && nargout <= 1, USAGE) ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;

    //--------------------------------------------------------------------------
    // serialize the matrix into the blob
    //--------------------------------------------------------------------------

    // TODO: get compression method from an input string

    void *blob = NULL ;
    size_t blob_size ;
    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;

    //--------------------------------------------------------------------------
    // free the shallow matrix A
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;

    //--------------------------------------------------------------------------
    // return the blob to MATLAB as a uint8 dense blobsize-by-1 array
    //--------------------------------------------------------------------------

    pargout [0] = mxCreateNumericMatrix (0, 1, mxUINT8_CLASS, mxREAL) ;
    mxFree (mxGetData (pargout [0])) ;
    mxSetData (pargout [0], blob) ;
    mxSetM (pargout [0], blob_size) ;
    GB_WRAPUP ;
}

