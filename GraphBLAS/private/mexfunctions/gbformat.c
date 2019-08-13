//------------------------------------------------------------------------------
// gbformat: get/set the matrix format to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

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

    gb_usage (nargin <= 1 && nargout <= 1,
        "usage: f = gb.format or gb.format (f)") ;

    //--------------------------------------------------------------------------
    // get/set the format
    //--------------------------------------------------------------------------

    GxB_Format_Value format ;

    if (nargin == 0)
    {

        //----------------------------------------------------------------------
        // format = gb.format
        //----------------------------------------------------------------------

        // get the global format
        GxB_Format_Value format ;
        OK (GxB_get (GxB_FORMAT, &format)) ;

    }
    else // if (nargin == 1)
    {

        if (mxIsChar (pargin [0]))
        {

            //------------------------------------------------------------------
            // gb.format (format)
            //------------------------------------------------------------------

            // set the global format
            format = gb_mxstring_to_format (pargin [0]) ;
            OK (GxB_set (GxB_FORMAT, format)) ;

        }
        else
        {

            //------------------------------------------------------------------
            // gb.format (G)
            //------------------------------------------------------------------

            // get the format of the input matrix G
            GrB_Matrix G = gb_get_shallow (pargin [0]) ;
            OK (GxB_get (G, GxB_FORMAT, &format)) ;
            OK (GrB_free (&G)) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (format == GxB_BY_ROW)
    {
        pargout [0] = mxCreateString ("by row") ;
    }
    else
    {
        pargout [0] = mxCreateString ("by col") ;
    }
}

