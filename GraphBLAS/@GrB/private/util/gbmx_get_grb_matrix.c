//------------------------------------------------------------------------------
// gbmx_get_grb_matrix: get a @GrB matrix argument
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
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
    // get the GrB_Matrix
    //--------------------------------------------------------------------------

    mxArray *G_opaque = gbmx_get_grb_handle (G) ;
    GrB_Matrix C = (*((GrB_Matrix *) mxGetData (G_opaque))) ;
    return (C) ;
}

