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
    GxB_Format_Value fmt ;
    GrB_Descriptor d = 
        gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt) ;

    if (d == NULL)
    { 
        printf ("\nDefault GraphBLAS descriptor:\n") ;
        OK (GrB_Descriptor_new (&d)) ;
    }

    OK (GxB_Descriptor_fprint (d, "", GxB_COMPLETE, NULL)) ;
    printf ("    d.kind     = ") ;

    switch (kind)
    {
        // for gb.extractuples:
        case KIND_0BASED : printf ("zero-based\n") ; break ;
        case KIND_1BASED : printf ("one-based\n") ;  break ;
        // for most gb.methods:
        case KIND_SPARSE : printf ("sparse\n") ;     break ;
        case KIND_FULL   : printf ("full\n") ;       break ;
        case KIND_GB     :
        default          : printf ("gb\n") ;         break ;
    }

    OK (GrB_free (&d)) ;
    GB_WRAPUP ;
}

