//------------------------------------------------------------------------------
// gb_typecast: typecast a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// TODO add option to create A as CSR/CSC, hypersparse/standard

#include "gb_matlab.h"

GrB_Matrix gb_typecast      // A = (type) S, where A is deep
(
    GrB_Type type,          // if NULL, copy but do not typecast
    GrB_Matrix S            // may be shallow or deep
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

        // OK (GxB_Matrix_fprint (S, "S to dup", 3, stdout)) ;
        OK (GrB_Matrix_dup (&A, S)) ;
        // OK (GxB_Matrix_fprint (A, "A to dupped", 3, stdout)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // typecast the input to the requested type
        //----------------------------------------------------------------------

        // OK (GxB_Matrix_fprint (S, "S to typecast", 3, stdout)) ;
        // OK (GxB_Type_fprint (type, "new type is:", 3, stdout)) ;
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
        // OK (GxB_Matrix_fprint (A, "A typecasted", 3, stdout)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    CHECK_ERROR (gb_is_shallow (A), "Hey, A shallow!") ;
    return (A) ;
}

