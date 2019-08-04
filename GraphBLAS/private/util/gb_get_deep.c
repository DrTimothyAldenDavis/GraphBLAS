//------------------------------------------------------------------------------
// gb_get_deep: create a deep GrB_Matrix copy of a MATLAB X
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

GrB_Matrix gb_get_deep      // return a deep GrB_Matrix copy of a MATLAB X
(
    const mxArray *X,       // input MATLAB matrix (sparse or struct)
    GrB_Type type           // typecast X to this type (NULL if no typecast)
)
{

    GrB_Matrix S = gb_get_shallow (X) ;
    GxB_Format_Value format ;
    OK (GxB_get (GxB_FORMAT, &format)) ;
    GrB_Matrix A = gb_typecast (type, format, S) ;
    OK (GrB_free (&S)) ;
    return (A) ;
}

