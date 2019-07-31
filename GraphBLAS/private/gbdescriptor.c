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

    bool kind_is_object ;
    GrB_Descriptor d = gb_mxarray_to_descriptor (pargin [0], &kind_is_object) ;

    if (d == NULL)
    {
        printf ("\nDefault GraphBLAS descriptor:\n") ;
        OK (GrB_Descriptor_new (&d)) ;
    }

    OK (GxB_Descriptor_fprint (d, "", GxB_COMPLETE, stdout)) ;
    printf ("d.kind = %s\n", (kind_is_object) ? "object" : "sparse") ;
    OK (GrB_free (&d)) ;
}

