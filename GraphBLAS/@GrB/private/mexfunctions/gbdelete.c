//------------------------------------------------------------------------------
// gbdelete: deletes a @GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbdelete (G)

// Deletes the @GrB matrix G.  Does nothing if the input is not a @GrB handle
// object from GraphBLAS v10.4.0 or later.  Since this is called by the @GrB
// delete method, this method must not throw an error (per the MATLAB
// specification of how handle objects are deleted).

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
//          bool pr = (*C_handle != NULL) ;
//          if (pr) printf ("gbdelete %p : %d to ", *C_handle,
//              GB_Global_nmalloc_get ( )) ;
            GrB_Matrix_free (C_handle) ;
//          if (pr) printf ("%d\n", GB_Global_nmalloc_get ( )) ;
        }
    }
}

