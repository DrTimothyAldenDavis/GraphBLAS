//------------------------------------------------------------------------------
// GzB_IndexBinaryOp_fprint: print and check a GzB_IndexBinaryOp object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GzB_IndexBinaryOp_fprint   // print and check a GzB_IndexBinaryOp
(
    GzB_IndexBinaryOp op,           // object to print and check
    const char *name,               // name of the object
    GxB_Print_Level pr,             // print level
    FILE *f                         // file for output
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GzB_IndexBinaryOp_fprint (op, name, pr, f)") ;

    //--------------------------------------------------------------------------
    // print and check the object
    //--------------------------------------------------------------------------

    return (GB_IndexBinaryOp_check (op, name, pr, f)) ;
}

