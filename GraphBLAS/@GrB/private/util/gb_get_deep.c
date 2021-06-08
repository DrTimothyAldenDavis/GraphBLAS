//------------------------------------------------------------------------------
// gb_get_deep: create a deep GrB_Matrix copy of a MATLAB X
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

#include "gb_matlab.h"

GrB_Matrix gb_get_deep      // return a deep GrB_Matrix copy of a MATLAB X
(
    const mxArray *X        // input MATLAB matrix (sparse or struct)
)
{ 

    GrB_Matrix S = gb_get_shallow (X) ;
// printf ("::::::::::::::::::::::::::::::::: Shallow S\n") ; GxB_print (S,3) ;
    GxB_Format_Value fmt ;
    OK (GxB_Matrix_Option_get (S, GxB_FORMAT, &fmt)) ;
    GrB_Matrix A = gb_typecast (S, NULL, fmt, 0) ;
// printf ("::::::::::::::::::::::::::::::::::: DEEP A\n") ; GxB_print (A,3) ;
    OK (GrB_Matrix_free (&S)) ;
    return (A) ;
}

