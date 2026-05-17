//------------------------------------------------------------------------------
// gbdelete: deletes a @GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbdelete (X)

// deletes the @GrB matrix G.  Does nothing if the input is not a @GrB matrix.

#include "gb_interface.h"

#define USAGE "usage: gbdelete (X)"

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

    gbmx_usage (nargin == 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the @GrB matrix handle to the GrB_Matrix, and free the matrix
    //--------------------------------------------------------------------------

    mxArray *G_opaque = gbmx_get_grb_handle (pargin [0]) ;
    GrB_Matrix *C_handle = (GrB_Matrix *) mxGetData (G_opaque) ;
    GrB_Matrix_free (C_handle) ;

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    gb_wrapup ( ) ;
}

