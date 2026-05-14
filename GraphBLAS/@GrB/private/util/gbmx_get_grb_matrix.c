//------------------------------------------------------------------------------
// gbmx_get_grb_matrix: get a @GrB matrix argument
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Matrix gbmx_get_grb_matrix  // the content of a MATLAB @GrB handle object
(
    // input
    const mxArray *G            // must be a @GrB object
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (G == NULL, "matrix missing") ;

    //--------------------------------------------------------------------------
    // get the GrB_Matrix
    //--------------------------------------------------------------------------

    mxArray *G_opaque = NULL ;
    if (mxIsStruct (G))
    { 
        G_opaque = mxGetFieldByNumber (G, 0, 0) ;
    }
    else if (mxIsClass (G, "GrB"))
    { 
        G_opaque = mxGetProperty (G, 0, "opaque") ;
    }

    CHECK_ERROR (G_opaque == NULL, "@GrB object corrupted") ;
    return (*((GrB_Matrix *) mxGetData (G_opaque))) ;
}

