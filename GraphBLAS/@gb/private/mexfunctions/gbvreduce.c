//------------------------------------------------------------------------------
// gbvreduce: reduce a matrix to a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// gbvreduce is an interface to GrB_Matrix_reduce.

// Usage:

//  Cout = gb.vreduce (op, A, desc)
//  Cout = gb.vreduce (Cin, M, op, A, desc)
//  Cout = gb.vreduce (Cin, accum, op, A, desc)
//  Cout = gb.vreduce (Cin, M, accum, op, A, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A and the descriptor).

#include "gb_matlab.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage ((nargin == 3 || nargin == 5 || nargin == 6) && nargout <= 1,
        "usage: Cout = gb.vreduce (Cin, M, accum, op, A, desc)") ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, M = NULL, A ;
    GrB_BinaryOp accum = NULL ;
    GrB_Monoid monoid = NULL ;
    GrB_Type atype, ctype ;

    kind_enum_t kind ;
    GrB_Descriptor desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind) ;

    if (nargin == 3)
    {

        //----------------------------------------------------------------------
        // Cout = gb.vreduce (op, A, desc)
        //----------------------------------------------------------------------

        A = gb_get_shallow (pargin [1]) ;
        OK (GxB_Matrix_type (&atype, A)) ;
        monoid = gb_mxstring_to_monoid (pargin [0], atype) ;

    }
    else if (nargin == 5 && mxIsChar (pargin [1]))
    {

        //----------------------------------------------------------------------
        // Cout = gb.vreduce (Cin, accum, op, A, desc)
        //----------------------------------------------------------------------

        C = gb_get_deep (pargin [0]) ;
        OK (GxB_Matrix_type (&ctype, C)) ;
        accum = gb_mxstring_to_binop (pargin [1], ctype) ;
        A = gb_get_shallow (pargin [3]) ;
        OK (GxB_Matrix_type (&atype, A)) ;
        monoid = gb_mxstring_to_monoid (pargin [2], atype) ;

    }
    else if (nargin == 5 && !mxIsChar (pargin [1]))
    {

        //----------------------------------------------------------------------
        // Cout = gb.vreduce (Cin, M, op, A, desc)
        //----------------------------------------------------------------------

        C = gb_get_deep (pargin [0]) ;
        M = gb_get_shallow (pargin [1]) ;
        A = gb_get_shallow (pargin [3]) ;
        OK (GxB_Matrix_type (&atype, A)) ;
        monoid = gb_mxstring_to_monoid (pargin [2], atype) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // Cout = gb.vreduce (Cin, M, accum, op, A, desc)
        //----------------------------------------------------------------------

        C = gb_get_deep (pargin [0]) ;
        OK (GxB_Matrix_type (&ctype, C)) ;
        M = gb_get_shallow (pargin [1]) ;
        accum = gb_mxstring_to_binop (pargin [2], ctype) ;
        A = gb_get_shallow (pargin [4]) ;
        OK (GxB_Matrix_type (&atype, A)) ;
        monoid = gb_mxstring_to_monoid (pargin [3], atype) ;

    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    {

        // get the descriptor contents to determine if A is transposed
        GrB_Desc_Value in0 ;
        OK (GxB_get (desc, GrB_INP0, &in0)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;

        // get the size of A
        GrB_Index anrows, ancols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        // determine the size of the vector C
        GrB_Index cnrows = (A_transpose) ? ancols : anrows ;

        // determine the type of C
        if (accum != NULL)
        {
            // if accum is present, use its ztype to determine the type of C
            OK (GxB_BinaryOp_ztype (&ctype, accum)) ;
        }
        else
        {
            // otherwise, use the ztype of the monoid as the type of C
            GrB_BinaryOp binop ;
            OK (GxB_Monoid_operator (&binop, monoid)) ;
            OK (GxB_BinaryOp_ztype (&ctype, binop)) ;
        }

        OK (GrB_Matrix_new (&C, ctype, cnrows, 1)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += reduce(A)
    //--------------------------------------------------------------------------

    OK (GrB_reduce (C, M, accum, monoid, A, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_free (&M)) ;
    OK (GrB_free (&A)) ;
    OK (GrB_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
}

