//------------------------------------------------------------------------------
// gbdeserialize: deserialize a blob into a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// gbdeserialize is an interface to GxB_Matrix_deserialize.

// Usage:

// A = gbdeserialize (blob)         % set the type of A from the blob
// A = gbdeserialize (blob, type)   % typecast to the given type 

#include "gb_interface.h"

#define USAGE "usage: A = GrB.deserialize (blob, type)"

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

    gb_usage ((nargin == 1 || nargin == 2) && nargout <= 1, USAGE) ;
    CHECK_ERROR (mxGetClassID (pargin [0]) != mxUINT8_CLASS
        || mxGetN (pargin [0]) != 1, "blob must be uint8 column vector") ;

    //--------------------------------------------------------------------------
    // get the blob and the optional type
    //--------------------------------------------------------------------------

    void *blob = mxGetData (pargin [0]) ;
    size_t blob_size = mxGetM (pargin [0]) ;
    GrB_Type ctype = (nargin > 1) ? gb_mxstring_to_type (pargin [1]) : NULL ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    OK (GxB_Matrix_deserialize (&C, blob, blob_size, ctype, NULL)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_GRB) ;
    GB_WRAPUP ;
}

