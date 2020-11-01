//------------------------------------------------------------------------------
// GxB_Vector_import_Bitmap: import a vector in bitmap format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Vector_import_Bitmap // import a bitmap vector
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    GrB_Index n,        // vector length
    GrB_Index nvals,    // # of entries in bitmap
    int8_t **vb,        // bitmap, size n
    void **vx,          // values, size n entries
    const GrB_Descriptor desc
)
{ 
GB_GOTCHA ;

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_import_Bitmap (&v, type, n, nvals,"
        " &vb, &vx, desc)") ;
    GB_BURBLE_START ("GxB_Vector_import_Bitmap") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // import the vector
    //--------------------------------------------------------------------------

    info = GB_import ((GrB_Matrix *) v, type, n, 1, 0, nvals, false, 0,
        NULL, NULL, vb, NULL, vx, GxB_BITMAP, true, desc) ;
    GB_BURBLE_END ;
    return (info) ;
}

