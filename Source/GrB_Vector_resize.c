//------------------------------------------------------------------------------
// GrB_Vector_resize: change the size of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Vector_resize      // change the size of a vector
(
    GrB_Vector w,               // vector to modify
    GrB_Index nrows_new         // new number of rows in vector
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GrB_Vector_resize (w, nrows_new)") ;
    GB_BURBLE_START ("GrB_Vector_resize") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;

    //--------------------------------------------------------------------------
    // resize the vector
    //--------------------------------------------------------------------------

    GrB_Info info = GB_resize ((GrB_Matrix) w, nrows_new, 1, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

