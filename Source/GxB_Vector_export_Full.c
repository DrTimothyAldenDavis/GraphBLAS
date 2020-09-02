//------------------------------------------------------------------------------
// GxB_Vector_export_Full: export a full vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Vector_export_Full   // export and free a full vector
(
    GrB_Vector *v,      // handle of vector to export and free
    GrB_Type *type,     // type of vector exported
    GrB_Index *n,       // length of the vector
    void **vx,          // values, size n
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_export_Full (&v, &type, &n, &vx, desc)") ;
    GB_BURBLE_START ("GxB_Vector_export_Full") ;
    GB_RETURN_IF_NULL (v) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (*v) ;
    if (!GB_is_dense (*v))
    { 
        // v must be dense or full
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // ensure the vector is full CSC
    //--------------------------------------------------------------------------

    ASSERT ((*v)->is_csc) ;
    GB_convert_any_to_full ((GrB_Matrix) *v) ;
    ASSERT (GB_IS_FULL (*v)) ;

    //--------------------------------------------------------------------------
    // export the vector
    //--------------------------------------------------------------------------

    GrB_Index vdim ;
    info = GB_export ((GrB_Matrix *) v, type, n, &vdim,
        NULL, NULL, NULL, NULL, NULL,
        NULL, NULL, NULL, NULL, vx, NULL, NULL, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

