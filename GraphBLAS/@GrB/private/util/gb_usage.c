//------------------------------------------------------------------------------
// gb_usage: check usage and make sure GrB.init has been called
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

void gb_usage       // check usage and make sure GrB.init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *message     // error message if usage is not correct
)
{

    //--------------------------------------------------------------------------
    // make sure GrB.init has been called
    //--------------------------------------------------------------------------

    if (!GB_Global_GrB_init_called_get ( ))
    {
        ERROR ("GrB.init must called before using any GraphBLAS function") ;
    }

    //--------------------------------------------------------------------------
    // check usage
    //--------------------------------------------------------------------------

    if (!ok)
    {
        ERROR (message) ;
    }

    //--------------------------------------------------------------------------
    // get test coverage
    //--------------------------------------------------------------------------

    #ifdef GBCOV
    gbcov_get ( ) ;
    #endif
}

