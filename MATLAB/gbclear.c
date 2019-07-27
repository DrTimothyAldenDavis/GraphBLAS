//------------------------------------------------------------------------------
// gbclear: clear all internal GraphBLAS workspace
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This function is optional for the MATLAB interface to GraphBLAS.  Simply
// terminating the MATLAB session, or typing 'clear all' will do the same
// thing.  However, if you are finished with GraphBLAS and wish to free its
// internal resources, then use this function.

#include "gbmex.h"

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

    gb_usage (nargin == 0 && nargout == 0, "usage: gbclear") ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    OK (GrB_finalize ( )) ;

    //--------------------------------------------------------------------------
    // allow a subsequent call to GxB_init
    //--------------------------------------------------------------------------

    GB_Global_GrB_init_called_set (false) ;
}

