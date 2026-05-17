//------------------------------------------------------------------------------
// gbmx_get_grb_handle: get a @GrB matrix from a struct/object, as a handle
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input to this method is an mxArray G, which must either be a @GrB
// object, or the G.opaque struct content of a @GrB object.  The output is
// a pointer to the GrB_Matrix that the @GrB object holds.

#include "gb_interface.h"

mxArray *gbmx_get_grb_handle    // the MATLAB @GrB opaque handle
(
    // input
    const mxArray *G            // must be a @GrB object
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (G == NULL, "matrix missing") ;
    mxArray *G_opaque = NULL ;

    //--------------------------------------------------------------------------
    // get the GrB_Matrix handle
    //--------------------------------------------------------------------------

    if (mxIsStruct (G))
    { 
        // G is a struct, which must come from the opaque content of a @GrB
        // object.  Results are undefined if G is another kind of struct.
        G_opaque = mxGetFieldByNumber (G, 0, 0) ;
    }
    else if (mxIsClass (G, "GrB"))
    { 
        // G is a @GrB object; get its opaque content (which must be a struct)
        // and then get the first item in the struct.
        mxArray *G_prop = mxGetProperty (G, 0, "opaque") ;
        CHECK_ERROR (!mxIsStruct (G_prop), "@GrB object corrupted") ;
        G_opaque = mxGetFieldByNumber (G_prop, 0, 0) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    CHECK_ERROR (G_opaque == NULL, "@GrB object corrupted") ;
    return (G_opaque) ;
}

