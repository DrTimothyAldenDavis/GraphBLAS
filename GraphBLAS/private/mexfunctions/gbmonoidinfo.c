//------------------------------------------------------------------------------
// gbmonoidinfo : print a GraphBLAS monoid (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Usage:

// gbmonoidinfo (monoid)
// gbmonoidinfo (monoid, type)

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

    gb_usage (nargin <= 2 && nargout == 0,
        "usage: gb.monoidinfo (monoid) or gb.monoidinfo (monoid,type)") ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS monoid and print it
    //--------------------------------------------------------------------------

    GrB_Type type = NULL ;
    if (nargin == 2)
    {
        type = gb_mxstring_to_type (pargin [1]) ;
        CHECK_ERROR (type == NULL, "unknown type") ;
    }

    GrB_Monoid op = gb_mxstring_to_monoid (pargin [0], type) ;
    OK (GxB_Monoid_fprint (op, "", GxB_COMPLETE, stdout)) ;
}

