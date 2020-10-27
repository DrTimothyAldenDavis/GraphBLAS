//------------------------------------------------------------------------------
// GxB_Vector_export_CSC: export a vector in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Vector_export_CSC  // export and free a CSC vector
(
    GrB_Vector *v,      // handle of vector to export and free
    GrB_Type *type,     // type of vector exported
    GrB_Index *n,       // length of the vector
    GrB_Index *nzmax,   // size of vi and vx
    GrB_Index *nvals,   // number of entries in the vector
    bool *jumbled,      // if true, indices may be unsorted
    GrB_Index **vi,     // indices, size nzmax
    void **vx,          // values, size nzmax entries
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_export_CSC (&v, &type, &n, &nzmax, &nvals,"
        " &jumbled, &vi, &vx, desc)") ;
    GB_BURBLE_START ("GxB_Vector_export_CSC") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;
    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_NULL (nvals) ;
    ASSERT_VECTOR_OK (*v, "v to export", GB0) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    if (jumbled == NULL)
    { 
        // the exported vector cannot be jumbled
        GB_MATRIX_WAIT (*v) ;
    }
    else
    {
        // the exported vector is allowed to be jumbled
        GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (*v) ;
    }

    //--------------------------------------------------------------------------
    // ensure the vector is sparse CSC
    //--------------------------------------------------------------------------

    ASSERT ((*v)->is_csc) ;
    GB_OK (GB_convert_any_to_sparse ((GrB_Matrix) *v, Context)) ;
    ASSERT (GB_IS_SPARSE (*v)) ;

    //--------------------------------------------------------------------------
    // export the vector
    //--------------------------------------------------------------------------

    int64_t nonempty, *vp = NULL ;
    GrB_Index vdim ;
    info = GB_export ((GrB_Matrix *) v, type, n, &vdim,
        nzmax, NULL, jumbled, &nonempty, NULL,
        &vp, NULL, NULL, vi, vx, NULL, NULL, Context) ;
    if (info == GrB_SUCCESS)
    {
        (*nvals) = vp [1] ;
    }
    GB_FREE (vp) ;
    GB_BURBLE_END ;
    return (info) ;
}

