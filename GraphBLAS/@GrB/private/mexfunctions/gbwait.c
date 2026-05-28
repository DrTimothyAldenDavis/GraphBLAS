//------------------------------------------------------------------------------
// gbwait: finish work in a @GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbwait (G)

// finishes all pending work in a @GrB matrix.  Does nothing if the input
// is not a @GrB matrix.

#include "gb_interface.h"

#define USAGE "usage: gbwait (G)"

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
    // wait on the matrix
    //--------------------------------------------------------------------------

    if (mxIsClass (pargin [0], "GrB"))
    { 
        GrB_Matrix A = gbmx_get_grb_matrix (pargin [0]) ;
        OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    }

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    gb_wrapup ( ) ;
}

