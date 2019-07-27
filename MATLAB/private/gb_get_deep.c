//------------------------------------------------------------------------------
// gb_get_deep: get a deep GrB_Matrix copy of a MATLAB X
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gbmex.h"

GrB_Matrix gb_get_deep      // return a deep GrB_Matrix copy of a MATLAB X
(
    const mxArray *X,       // input MATLAB matrix (sparse or struct)
    GrB_Type type           // typecast X to this type (NULL if no typecast)
)
{

    GrB_Matrix S = gb_get_shallow (X) ;
    GrB_Matrix A = gb_typecast (type, S) ;
    gb_free_shallow (&S) ;
    return (A) ;
}
