//------------------------------------------------------------------------------
// gb_typecast: typecast a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

GrB_Matrix gb_typecast      // A = (type) S, where A is deep
(
    GrB_Type type,              // if NULL, copy but do not typecast
    GxB_Format_Value format,    // also convert to the requested format
    GrB_Matrix S                // may be shallow
)
{

    GrB_Matrix A ;

    if (S == NULL)
    {

        //----------------------------------------------------------------------
        // S is null; nothing to do
        //----------------------------------------------------------------------

        // This is not an error since the input matrix may be optional.
        A = NULL ;

    }
    else if (type == NULL)
    {

        //----------------------------------------------------------------------
        // make a deep copy of the input
        //----------------------------------------------------------------------

        OK (GrB_Matrix_dup (&A, S)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // typecast the input to the requested type
        //----------------------------------------------------------------------

        GrB_Index nrows, ncols ;
        OK (GrB_Matrix_nrows (&nrows, S)) ;
        OK (GrB_Matrix_ncols (&ncols, S)) ;
        OK (GrB_Matrix_new (&A, type, nrows, ncols)) ;

        // create a descriptor with d.trans = transpose
        GrB_Descriptor d ;
        OK (GrB_Descriptor_new (&d)) ;
        OK (GrB_Descriptor_set (d, GrB_INP0, GrB_TRAN)) ;

        // A = (type) S
        OK (GrB_transpose (A, NULL, NULL, S, d)) ;

        OK (GrB_free (&d)) ;
    }

    //--------------------------------------------------------------------------
    // convert the matrix to the right format
    //--------------------------------------------------------------------------

    OK (GxB_set (A, GxB_FORMAT, format)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (A) ;
}

