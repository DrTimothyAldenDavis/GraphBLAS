//------------------------------------------------------------------------------
// gbsemiring: create a GraphBLAS semiring and print it (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Usage:

// A = gbsemiring (semiring_string)
// A = gbsemiring (semiring_string, type)

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
        "usage: gbsemiring (semiring) or gbsemiring (semiring,type)") ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS semiring and print it
    //--------------------------------------------------------------------------

    GrB_Type type = NULL ;
    if (nargin == 2)
    {
        type = gb_mxstring_to_type (pargin [1]) ;
    }

    GrB_Semiring semiring = gb_mxstring_to_semiring (pargin [0], type) ;
    GxB_print (semiring, GxB_COMPLETE) ;
    GrB_free (&semiring) ;
}

