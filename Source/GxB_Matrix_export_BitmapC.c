//------------------------------------------------------------------------------
// GxB_Matrix_export_BitmapC: export a bitmap matrix, held by column
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_export_BitmapC  // export and free a bitmap matrix, by col
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix exported
    GrB_Index *nrows,   // number of rows of the matrix
    GrB_Index *ncols,   // number of columns of the matrix
    GrB_Index *nvals,   // # of entries
    int8_t **Ab,        // bitmap, size nrows*ncols
    void **Ax,          // values, size nrows*ncols entries
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_export_BitmapC (&A, &type, &nrows, &ncols, &nvals,"
        " &Ab, &Ax, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_export_BitmapC") ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*A) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (*A) ;

    //--------------------------------------------------------------------------
    // ensure the matrix is bitmap CSC
    //--------------------------------------------------------------------------

    // ensure the matrix is in CSC format
    if (!((*A)->is_csc))
    { 
        // A = A', done in-place, to put A in CSC format
        GBURBLE ("(transpose) ") ;
        GB_OK (GB_transpose (NULL, NULL, true, *A,
            NULL, NULL, NULL, false, Context)) ;
    }

    GB_OK (GB_convert_any_to_bitmap (*A, Context)) ;
    ASSERT (GB_IS_BITMAP (*A)) ;

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    int sparsity ;
    bool is_csc ;
    info = GB_export (A, type, nrows, ncols, NULL, nvals, NULL, NULL,
        NULL, NULL, Ab, NULL, Ax, &sparsity, &is_csc, Context) ;
    if (info == GrB_SUCCESS)
    {
        ASSERT (sparsity == GxB_BITMAP) ;
        ASSERT (is_csc) ;
    }
    GB_BURBLE_END ;
    return (info) ;
}

