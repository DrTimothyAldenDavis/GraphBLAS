//------------------------------------------------------------------------------
// gbselectopinfo : print a GraphBLAS selectop (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// Usage:

// gbselectopinfo (selectop)

#include "gb_interface.h"

#define USAGE "usage: GrB.selectopinfo (selectop)"

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

    gb_usage (nargin == 1 && nargout == 0, USAGE) ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS selectop and print it
    //--------------------------------------------------------------------------

    #define LEN 256
    char opstring [LEN+2] ;
    gb_mxstring_to_string (opstring, LEN, pargin [0], "select operator") ;

    GxB_SelectOp op = gb_mxstring_to_selectop (pargin [0]) ;
    OK (GxB_SelectOp_fprint (op, opstring, GxB_COMPLETE, NULL)) ;
    GB_WRAPUP ;
}

