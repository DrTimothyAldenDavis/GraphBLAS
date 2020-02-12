//------------------------------------------------------------------------------
// gbsetup: initialize or finalize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The gbsetup mexFunction is the only mexFunction that locks itself into
// MATLAB working memory.  gbsetup ('start') locks this mexFunction, and then
// initializes GraphBLAS by setting all GraphBLAS global variables and calling
// GxB_init.  gbsetup ('finish') unlocks this mexFunction, and finalizes
// GraphBLAS by calling GrB_finalize.

// Usage:

// gbsetup ('start') ;
// gbsetup ('finish') ;

#include "gb_matlab.h"
#include "GB_printf.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // get test coverage
    //--------------------------------------------------------------------------

    #ifdef GBCOV
    gbcov_get ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // register the function to clear GraphBLAS
    //--------------------------------------------------------------------------

    mexAtExit (gb_at_exit) ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (nargin != 1 || nargout != 0 || !mxIsChar (pargin [0]))
    {
        ERROR ("usage: gbsetup (action)") ;
    }

    //--------------------------------------------------------------------------
    // get the action
    //--------------------------------------------------------------------------

    #define LEN 256
    char action [LEN+2] ;
    gb_mxstring_to_string (action, LEN, pargin [0], "action") ;

    if (MATCH (action, "start"))
    { 

        //----------------------------------------------------------------------
        // initialize GraphBLAS
        //----------------------------------------------------------------------

        if (mexIsLocked ( ) || GB_Global_GrB_init_called_get ( ))
        {
            ERROR ("GrB.init already called") ;
        }
        mexLock ( ) ;

        //----------------------------------------------------------------------
        // set the printf function
        //----------------------------------------------------------------------

        GB_printf_function = mexPrintf ;

        //----------------------------------------------------------------------
        // initialize GraphBLAS
        //----------------------------------------------------------------------

        OK (GxB_init (GrB_NONBLOCKING, mxMalloc, mxCalloc, mxRealloc, mxFree,
            false)) ;

        //----------------------------------------------------------------------
        // MATLAB matrices are stored by column
        //----------------------------------------------------------------------

        OK (GxB_Global_Option_set (GxB_FORMAT, GxB_BY_COL)) ;

        // print short format by default
        GB_Global_print_format_set (1) ;

        // print 1-based indices
        GB_Global_print_one_based_set (true) ;

    }
    else if (MATCH (action, "finish"))
    { 

        //----------------------------------------------------------------------
        // finalize GraphBLAS
        //----------------------------------------------------------------------

        if (!mexIsLocked ( ))
        {
            ERROR ("GrB.finalize can only be called after GrB.init") ;
        }
        mexUnlock ( ) ;

        gb_at_exit ( ) ;

    }
    else
    { 
        ERROR ("gbsetup: unknown action") ;
    }

    //--------------------------------------------------------------------------
    // save test coverage
    //--------------------------------------------------------------------------

    GB_WRAPUP ;
}

