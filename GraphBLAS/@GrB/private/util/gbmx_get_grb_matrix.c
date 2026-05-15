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
        // printf ("get G struct\n") ;
        G_opaque = mxGetFieldByNumber (G, 0, 0) ;
    }
    else if (mxIsClass (G, "GrB"))
    { 
//      printf ("get G object opaque\n") ;
        mxArray *G_prop = mxGetProperty (G, 0, "opaque") ;
//      printf ("G_prop is struct: %d\n", mxIsStruct (G_prop)) ;
//      printf ("got G_prop %p\n", G_prop) ;
        CHECK_ERROR (!mxIsStruct (G_prop), "@GrB object corrupted 1") ;
        G_opaque = mxGetFieldByNumber (G_prop, 0, 0) ;
//      printf ("got G_opaque %p\n", G_opaque) ;
    }

    CHECK_ERROR (G_opaque == NULL, "@GrB object corrupted 2") ;
    GrB_Matrix C = (*((GrB_Matrix *) mxGetData (G_opaque))) ;
    // printf ("got C header is %p\n", C) ;
    return (C) ;
}

