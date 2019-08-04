//------------------------------------------------------------------------------
// gbdescriptorinfo: print a GraphBLAS descriptor (for illustration only)
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

    gb_usage (nargin <= 1 && nargout == 0,
        "usage: gb.descriptorinfo or gb.descriptorinfo (d)") ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS descriptor and print it
    //--------------------------------------------------------------------------

    kind_enum_t kind ;
    GrB_Descriptor d = gb_mxarray_to_descriptor (pargin [0], &kind) ;

    if (d == NULL)
    {
        printf ("\nDefault GraphBLAS descriptor:\n") ;
        OK (GrB_Descriptor_new (&d)) ;
    }

    OK (GxB_Descriptor_fprint (d, "", GxB_COMPLETE, stdout)) ;

    switch (kind)
    {
        case KIND_SPARSE: printf ("d.kind     = sparse\n") ;    break ;
        case KIND_FULL:   printf ("d.kind     = full\n") ;      break ;
        case KIND_GB:
        default:          printf ("d.kind     = gb\n") ;        break ;
    }

    OK (GrB_free (&d)) ;
}

