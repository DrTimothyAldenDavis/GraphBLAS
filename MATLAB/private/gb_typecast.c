//------------------------------------------------------------------------------
// gb_typecast: typecast a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gbmex.h"

GrB_Matrix gb_typecast      // A = (type) S, where A is deep
(
    GrB_Type type,          // if NULL, copy but do not typecast
    GrB_Matrix S            // may be shallow or deep
)
{

    GrB_Matrix A ;

    if (type == NULL)
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

        // TODO: create a set of descriptors in gbinit.c
        GrB_Descriptor d ;
        OK (GrB_Descriptor_new (&d)) ;
        OK (GrB_Descriptor_set (d, GrB_INP0, GrB_TRAN)) ;

        // A = (type) S
        OK (GrB_transpose (A, NULL, NULL, S, d)) ;

        OK (GrB_Descriptor_free (&d)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (A) ;
}
