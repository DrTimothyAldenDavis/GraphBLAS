//------------------------------------------------------------------------------
// GxB_Vector_import_Full: import a vector in full format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Vector_import_Full // import a full vector
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    GrB_Index n,        // vector length
    void **vx,          // values, size n entries
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_import_Full (&v, type, n, &vx, desc)") ;
    GB_BURBLE_START ("GxB_Vector_import_Full") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // import the vector
    //--------------------------------------------------------------------------

    info = GB_import ((GrB_Matrix *) v, type, n, 1, 0, 0, false, 0, 0,
        NULL, NULL, NULL, NULL, vx, GxB_FULL, true, desc) ;
    GB_BURBLE_END ;
    return (info) ;
}

