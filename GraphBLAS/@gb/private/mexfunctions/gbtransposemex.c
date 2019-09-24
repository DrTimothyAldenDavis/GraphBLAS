//------------------------------------------------------------------------------
// gbtransposemex: sparse matrix transpose
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// gbtranpose is an interface to GrB_transpose.

// Usage:

// Cout = gb.gbtranspose (A, desc)
// Cout = gb.gbtranspose (Cin, accum, A, desc)
// Cout = gb.gbtranspose (Cin, M, A, desc)
// Cout = gb.gbtranspose (Cin, M, accum, A, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A and the descriptor).  Note that if desc.in0
// is 'transpose', then C<M>=A or C<M>+=A is computed, with A not transposed,
// since the default behavior is to transpose the input matrix.

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

    gb_usage ((nargin == 2 || nargin == 4 || nargin == 5) && nargout <= 1,
        "usage: Cout = gb.gbtranspose (Cin, M, accum, A, desc)") ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, M = NULL, A ;
    GrB_BinaryOp accum = NULL ;
    GrB_Type ctype ;

    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    GrB_Descriptor desc = 
        gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt) ;

    if (nargin == 2)
    { 

        //----------------------------------------------------------------------
        // Cout = gb.gbtranspose (A, desc)
        //----------------------------------------------------------------------

        A = gb_get_shallow (pargin [0]) ;

    }
    else if (nargin == 4 && mxIsChar (pargin [1]))
    { 

        //----------------------------------------------------------------------
        // Cout = gb.gbtranspose (Cin, accum, A, desc)
        //----------------------------------------------------------------------

        C = gb_get_deep (pargin [0]) ;
        OK (GxB_Matrix_type (&ctype, C)) ;
        accum = gb_mxstring_to_binop (pargin [1], ctype) ;
        A = gb_get_shallow (pargin [2]) ;

    }
    else if (nargin == 4 && !mxIsChar (pargin [1]))
    { 

        //----------------------------------------------------------------------
        // Cout = gb.gbtranspose (Cin, M, A, desc)
        //----------------------------------------------------------------------

        C = gb_get_deep (pargin [0]) ;
        M = gb_get_shallow (pargin [1]) ;
        A = gb_get_shallow (pargin [2]) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // Cout = gb.gbtranspose (Cin, M, accum, A, desc)
        //----------------------------------------------------------------------

        C = gb_get_deep (pargin [0]) ;
        OK (GxB_Matrix_type (&ctype, C)) ;
        M = gb_get_shallow (pargin [1]) ;
        accum = gb_mxstring_to_binop (pargin [2], ctype) ;
        A = gb_get_shallow (pargin [3]) ;

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

        // determine the size of C
        GrB_Index cnrows = (A_transpose) ? anrows : ancols ;
        GrB_Index cncols = (A_transpose) ? ancols : anrows ;

        // use the type of A
        OK (GxB_Matrix_type (&ctype, A)) ;

        OK (GrB_Matrix_new (&C, ctype, cnrows, cncols)) ;
        fmt = gb_get_format (cnrows, cncols, A, NULL, fmt) ;
        OK (GxB_set (C, GxB_FORMAT, fmt)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += A or A'
    //--------------------------------------------------------------------------

    OK (GrB_transpose (C, M, accum, A, desc)) ;

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
    GB_WRAPUP ;
}

