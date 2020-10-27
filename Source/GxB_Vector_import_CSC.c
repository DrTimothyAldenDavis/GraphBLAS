//------------------------------------------------------------------------------
// GxB_Vector_import_CSC: import a vector in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

GrB_Info GxB_Vector_import_CSC  // import a vector in CSC format
(
    GrB_Vector *v,      // handle of vector to create
    GrB_Type type,      // type of vector to create
    GrB_Index n,        // vector length
    GrB_Index nzmax,    // size of vi and vx
    GrB_Index nvals,    // # of entries in vector
    bool jumbled,       // if true, indices may be unsorted
    GrB_Index **vi,     // indices, size nzmax
    void **vx,          // values, size nzmax entries
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_import_CSC (&v, type, n, nzmax, nvals, jumbled,"
        " &vi, &vx, desc)") ;
    GB_BURBLE_START ("GxB_Vector_import_CSC") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // import the vector
    //--------------------------------------------------------------------------

    GrB_Index *vp = GB_MALLOC (2, int64_t) ;
    if (vp == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    vp [0] = 0 ;
    vp [1] = nvals ;
    int64_t nonempty = (nvals > 0) ;
    info = GB_import ((GrB_Matrix *) v, type, n, 1, nzmax, 0, jumbled,
        nonempty, 0, &vp, NULL, NULL, vi, vx, GxB_SPARSE, true, desc) ;
    if (info != GrB_SUCCESS)
    { 
        GB_FREE (vp) ;
    }
    GB_BURBLE_END ;
    return (info) ;
}

