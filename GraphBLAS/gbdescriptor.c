//------------------------------------------------------------------------------
// gbdescriptor: create a GraphBLAS descriptor and print it (for illustration)
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

    gb_usage (nargin <= 1 && nargout == 0, "usage: gbdescriptor (d)") ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS descriptor and print it
    //--------------------------------------------------------------------------

    GrB_Descriptor d = gb_mxarray_to_descriptor (pargin [0]) ;

    if (d == NULL)
    {
        printf ("\nDefault GraphBLAS descriptor:\n") ;
        OK (GrB_Descriptor_new (&d)) ;
    }

    GxB_print (d, GxB_COMPLETE) ;
    GrB_free (&d) ;
}

