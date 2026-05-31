//------------------------------------------------------------------------------
// gbtype: type of a GraphBLAS matrix struct, or any built-in variable
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be any built-in variable.  If it is a GraphBLAS G.opaque
// struct, then its internal type is returned.

// Usage

// type = gbtype (X)

// Calls to GrB_* and mx* methods are intermingled since none of the GrB
// methods allocate any memory.

#include "gb_interface.h"

#define USAGE "usage: type = gbtype (X)"

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

    gbmx_usage (nargin == 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the type of the matrix
    //--------------------------------------------------------------------------

    if (mxIsStruct (pargin [0]) || mxIsClass (pargin [0], "GrB"))
    { 
        // get the type of a @GrB matrix
        GrB_Type type ;
        GrB_Matrix A = gbmx_get_grb_matrix (pargin [0]) ;
        CHECK_ERROR (A == NULL, "invalid @GrB matrix") ;
        OK (GxB_Matrix_type (&type, A)) ;
        pargout [0] = gbmx_type_to_mxstring (type) ;
    }
    else
    { 
        // get the type of a MATLAB matrix
        mxClassID class = mxGetClassID (pargin [0]) ;
        bool is_complex = mxIsComplex (pargin [0]) ;
        pargout [0] = gbmx_mxclass_to_mxstring (class, is_complex) ;
    }

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    gb_wrapup ( ) ;
}

