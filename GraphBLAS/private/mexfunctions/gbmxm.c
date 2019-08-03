//------------------------------------------------------------------------------
// gbmxm: sparse matrix-matrix multiplication
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// TODO add desc.format = 'csr', 'csc', 'hcsr', 'hcsc', 'matlab', ...

// gbmxm is an interface to GrB_mxm.

// Usage:

// Cout = gbmxm (semiring, A, B)
// Cout = gbmxm (semiring, A, B, desc)

// Cout = gbmxm (Cin, accum, semiring, A, B)
// Cout = gbmxm (Cin, accum, semiring, A, B, desc)

// Cout = gbmxm (Cin, Mask, semiring, A, B)
// Cout = gbmxm (Cin, Mask, semiring, A, B, desc)

// Cout = gbmxm (Cin, Mask, accum, semiring, A, B)
// Cout = gbmxm (Cin, Mask, accum, semiring, A, B, desc)

// If Cin is not present or is an empty matrix (Cin = [ ]) then it is
// implicitly a matrix with no entries, of the right size (which depends on A,
// B, and the descriptor).

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

    gb_usage (nargin >= 3 && nargin <= 7 && nargout <= 1,
        "usage: Cout = gb.mxm (Cin, Mask, accum, semiring, A, B, desc)") ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, M = NULL, A, B ;
    GrB_BinaryOp accum = NULL, add = NULL ;
    GrB_Semiring semiring ;
    GrB_Type atype, ctype ;
    GrB_Descriptor desc = NULL ;
    bool kind_is_object = false ;

    if (mxIsChar (pargin [0]))
    {
        // GB_HERE ;

        //----------------------------------------------------------------------
        // Cout = gbmxm (semiring, A, B)
        // Cout = gbmxm (semiring, A, B, desc)
        //----------------------------------------------------------------------

        gb_usage (nargin == 3 || nargin == 4,
            "usage: Cout = gb.mxm (semiring, A, B, desc)") ;

        A = gb_get_shallow (pargin [1]) ;
        B = gb_get_shallow (pargin [2]) ;
        if (nargin > 3)
        {
            desc = gb_mxarray_to_descriptor (pargin [3], &kind_is_object) ;
        }
        OK (GxB_Matrix_type (&atype, A)) ;
        semiring = gb_mxstring_to_semiring (pargin [0], atype) ;
        // GB_HERE ;

    }
    else if (mxIsChar (pargin [1]) && mxIsChar (pargin [2]))
    {

        //----------------------------------------------------------------------
        // Cout = gbmxm (Cin, accum, semiring, A, B)
        // Cout = gbmxm (Cin, accum, semiring, A, B, desc)
        //----------------------------------------------------------------------

        gb_usage (nargin == 5 || nargin == 6,
            "usage: Cout = gb.mxm (Cin, accum, semiring, A, B, desc)") ;

        C = gb_get_deep (pargin [0], NULL) ;
        OK (GxB_Matrix_type (&ctype, C)) ;
        accum = gb_mxstring_to_binop (pargin [1], ctype) ;
        A = gb_get_shallow (pargin [3]) ;
        B = gb_get_shallow (pargin [4]) ;
        if (nargin > 5)
        {
            desc = gb_mxarray_to_descriptor (pargin [5], &kind_is_object) ;
        }
        OK (GxB_Matrix_type (&atype, A)) ;
        semiring = gb_mxstring_to_semiring (pargin [2], atype) ;

    }
    else if (mxIsChar (pargin [2]) && !mxIsChar (pargin [3]))
    {

        //----------------------------------------------------------------------
        // Cout = gbmxm (Cin, Mask, semiring, A, B)
        // Cout = gbmxm (Cin, Mask, semiring, A, B, desc)
        //----------------------------------------------------------------------

        gb_usage (nargin == 5 || nargin == 6,
            "usage: Cout = gb.mxm (Cin, Mask, semiring, A, B, desc)") ;

        C = gb_get_deep (pargin [0], NULL) ;
        M = gb_get_shallow (pargin [1]) ;
        A = gb_get_shallow (pargin [3]) ;
        B = gb_get_shallow (pargin [4]) ;
        if (nargin > 5)
        {
            desc = gb_mxarray_to_descriptor (pargin [5], &kind_is_object) ;
        }
        OK (GxB_Matrix_type (&atype, A)) ;
        semiring = gb_mxstring_to_semiring (pargin [2], atype) ;

    }
    else if (mxIsChar (pargin [2]) && mxIsChar (pargin [3]))
    {

        //----------------------------------------------------------------------
        // Cout = gbmxm (Cin, Mask, accum, semiring, A, B)
        // Cout = gbmxm (Cin, Mask, accum, semiring, A, B, desc)
        //----------------------------------------------------------------------

        gb_usage (nargin == 6 || nargin == 7,
            "usage: Cout = gb.mxm (Cin, Mask, accum, semiring, A, B, desc)") ;

        C = gb_get_deep (pargin [0], NULL) ;
        OK (GxB_Matrix_type (&ctype, C)) ;
        M = gb_get_shallow (pargin [1]) ;
        accum = gb_mxstring_to_binop (pargin [2], ctype) ;
        A = gb_get_shallow (pargin [4]) ;
        B = gb_get_shallow (pargin [5]) ;
        if (nargin > 6)
        {
            desc = gb_mxarray_to_descriptor (pargin [6], &kind_is_object) ;
        }
        OK (GxB_Matrix_type (&atype, A)) ;
        semiring = gb_mxstring_to_semiring (pargin [3], atype) ;

    }
    else
    {
        USAGE ("Cout = gbmxm (Cin, Mask, accum, semiring, A, B, desc)") ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is either not present on input, or present but
    // passed in as Cin = [ ].  Construct C of the right size and type.

    if (C == NULL)
    {

        // get the descriptor contents to determine if A and B are transposed
        GrB_Desc_Value in0, in1 ;
        OK (GxB_get (desc, GrB_INP0, &in0)) ;
        OK (GxB_get (desc, GrB_INP1, &in1)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;
        bool B_transpose = (in1 == GrB_TRAN) ;

        // get the size of A and B
        GrB_Index anrows, ancols, bnrows, bncols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;
        OK (GrB_Matrix_nrows (&bnrows, B)) ;
        OK (GrB_Matrix_ncols (&bncols, B)) ;

        // determine the size of C
        GrB_Index cnrows = (A_transpose) ? ancols : anrows ;
        GrB_Index cncols = (B_transpose) ? bnrows : bncols ;

        // determine the type of C
        if (accum != NULL)
        {
            // if accum is present, use its ztype to determine the type of C
            OK (GxB_BinaryOp_ztype (&ctype, accum)) ;
        }
        else
        {
            // otherwise, use the semiring's additive monoid as the type of C
            GrB_Monoid add_monoid ;
            OK (GxB_Semiring_add (&add_monoid, semiring)) ;
            OK (GxB_Monoid_operator (&add, add_monoid)) ;
            OK (GxB_BinaryOp_ztype (&ctype, add)) ;
        }

        // TODO all the user to determine the CSR/CSC format
        OK (GrB_Matrix_new (&C, ctype, cnrows, cncols)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += A*B
    //--------------------------------------------------------------------------

    // OK (GxB_Matrix_fprint (C, "C input", 3, stdout)) ;
    // OK (GxB_Matrix_fprint (A, "A input", 3, stdout)) ;
    // OK (GxB_Matrix_fprint (B, "B input", 3, stdout)) ;
    // if (desc != NULL)
        // OK (GxB_Descriptor_fprint (desc, "desc input", 3, stdout)) ;
    // OK (GxB_Semiring_fprint (semiring, "semiring input", 3, stdout)) ;
    // if (accum != NULL)
        // OK (GxB_BinaryOp_fprint (accum, "accum input", 3, stdout)) ;
    // if (M != NULL)
        // OK (GxB_Matrix_fprint (M, "M input", 3, stdout)) ;
        // GB_HERE ;
    OK (GrB_mxm (C, M, accum, semiring, A, B, desc)) ;
        // GB_HERE ;

    // OK (GxB_Matrix_fprint (C, "C from mxm", 3, stdout)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_free (&M)) ;
    OK (GrB_free (&A)) ;
    OK (GrB_free (&B)) ;
    OK (GrB_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind_is_object) ;

    // TODO: hack because Sauna is freed by MATLAB.  Must make it persistent...
    gb_at_exit ( ) ;

}

