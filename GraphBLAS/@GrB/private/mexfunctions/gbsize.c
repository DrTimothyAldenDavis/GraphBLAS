//------------------------------------------------------------------------------
// gbsize: dimension and type of a GraphBLAS or built-in matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS @GrB matrix or a standard built-in
// matrix.  Note that the output is int64, to accomodate huge hypersparse
// matrices.  Also returns the type of the matrix.

// Usage:

// [m, n, type] = gbsize (X)

// Calls to GrB_* and mx* methods are intermingled since none of the GrB
// methods allocate any memory.

#include "gb_interface.h"

#define USAGE "usage: [m n type] = gbsize (X)"

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

    gbmx_usage (nargin == 1 && nargout <= 4, USAGE) ;

    //--------------------------------------------------------------------------
    // get the # of rows and columns of a GraphBLAS or built-in matrix
    //--------------------------------------------------------------------------

    uint64_t nrows, ncols ;

    if (mxIsStruct (pargin [0]) || mxIsClass (pargin [0], "GrB"))
    { 

        //----------------------------------------------------------------------
        // get the size of a GraphBLAS matrix
        //----------------------------------------------------------------------

        GrB_Matrix A = gbmx_get_grb_matrix (pargin [0]) ;
        CHECK_ERROR (A == NULL, "invalid @GrB matrix") ;
        OK (GrB_Matrix_nrows (&nrows, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;

        //----------------------------------------------------------------------
        // return type of a GraphBLAS matrix, if requested
        //----------------------------------------------------------------------

        if (nargout > 2)
        { 
            // return the type
            GrB_Type type ;
            OK (GxB_Matrix_type (&type, A)) ;
            pargout [2] = gbmx_type_to_mxstring (type) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // get the size of a built-in matrix
        //----------------------------------------------------------------------

        nrows = (uint64_t) mxGetM (pargin [0]) ;
        ncols = (uint64_t) mxGetN (pargin [0]) ;

        //----------------------------------------------------------------------
        // get the type of a built-in matrix, if requested
        //----------------------------------------------------------------------

        if (nargout > 2)
        { 
            mxClassID class = mxGetClassID (pargin [0]) ;
            bool is_complex = mxIsComplex (pargin [0]) ;
            pargout [2] = gbmx_mxclass_to_mxstring (class, is_complex) ;
        }
    }

    //--------------------------------------------------------------------------
    // return the size as int64 or double
    //--------------------------------------------------------------------------

    if (nrows > FLINTMAX || ncols > FLINTMAX)
    { 
        // output is int64 to avoid flint overflow
        int64_t *p ;
        pargout [0] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        // use mxGetData (best for Octave, fine for MATLAB)
        p = (int64_t *) mxGetData (pargout [0]) ;
        p [0] = (int64_t) nrows ;
        pargout [1] = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        p = (int64_t *) mxGetData (pargout [1]) ;
        p [0] = (int64_t) ncols ;
    }
    else
    { 
        // output is double
        pargout [0] = mxCreateDoubleScalar ((double) nrows) ;
        pargout [1] = mxCreateDoubleScalar ((double) ncols) ;
    }
}
