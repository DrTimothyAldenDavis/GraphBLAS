//------------------------------------------------------------------------------
// gbdelete: deletes a @GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbdelete (G)

// Deletes the @GrB matrix G.  Does nothing if the input is not a @GrB handle
// object from GraphBLAS v10.4.0 or later.  This method must not throw an
// error, since it is called by the @GrB delete method.

#include "gb_interface.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // get the @GrB matrix handle to the GrB_Matrix, and free the matrix
    //--------------------------------------------------------------------------

    if (nargin == 1)
    {
        mxArray *G_opaque = gbmx_get_grb_handle (pargin [0]) ;
        if (G_opaque != NULL)
        {
            GrB_Matrix *C_handle = (GrB_Matrix *) mxGetData (G_opaque) ;
            GrB_Matrix_free (C_handle) ;
        }
    }
}

