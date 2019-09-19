//------------------------------------------------------------------------------
// gbselect: select entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// gbselect is an interface to GxB_select.

// Usage:

// Cout = gbselect (op, A, desc)
// Cout = gbselect (op, A, thunk, desc)

// Cout = gbselect (Cin, accum, op, A, desc)
// Cout = gbselect (Cin, accum, op, A, thunk, desc)

// Cout = gbselect (Cin, M, op, A, desc)
// Cout = gbselect (Cin, M, op, A, thunk, desc)

// Cout = gbselect (Cin, M, accum, op, A, desc)
// Cout = gbselect (Cin, M, accum, op, A, thunk, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A, and the descriptor).  The type if Cin, if
// not present, is determined by the ztype of the accum, if present, or
// otherwise it has the same time as A.

// If op is 'eqthunk' or 'nethunk' and thunk is a NaN, and A has type GrB_FP32
// or GrB_FP64, then a user-defined operator is used instead of GxB_EQ_THUNK or
// GxB_NE_THUNK.

#include "gb_matlab.h"

bool gb_isnan32 (GrB_Index i, GrB_Index j, GrB_Index nrows, GrB_Index ncols,
    const void *x, const void *thunk)
{
    float aij = * ((float *) x) ;
    return (isnan (aij)) ;
}

bool gb_isnan64 (GrB_Index i, GrB_Index j, GrB_Index nrows, GrB_Index ncols,
    const void *x, const void *thunk)
{
    double aij = * ((double *) x) ;
    return (isnan (aij)) ;
}

bool gb_isnotnan32 (GrB_Index i, GrB_Index j, GrB_Index nrows, GrB_Index ncols,
    const void *x, const void *thunk)
{
    float aij = * ((float *) x) ;
    return (!isnan (aij)) ;
}

bool gb_isnotnan64 (GrB_Index i, GrB_Index j, GrB_Index nrows, GrB_Index ncols,
    const void *x, const void *thunk)
{
    double aij = * ((double *) x) ;
    return (!isnan (aij)) ;
}

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
        "usage: Cout = gb.select (Cin, M, accum, op, A, thunk, desc)") ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, M = NULL, A, thunk = NULL ;
    GrB_BinaryOp accum = NULL ;
    GxB_SelectOp op = NULL ;
    GrB_Type ctype ;

    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    GrB_Descriptor desc = 
        gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt) ;

    if (mxIsChar (pargin [0]))
    {

        //----------------------------------------------------------------------
        // Cout = gbselect (op, A, desc)
        // Cout = gbselect (op, A, thunk, desc)
        //----------------------------------------------------------------------

        gb_usage (nargin == 3 || nargin == 4,
            "usage: Cout = gb.select (op, A, thunk, desc)") ;

        op = gb_mxstring_to_selectop (pargin [0]) ;
        A = gb_get_shallow (pargin [1]) ;
        thunk = (nargin > 3) ? (GxB_Scalar) gb_get_shallow (pargin [2]) : NULL ;

    }
    else if (mxIsChar (pargin [1]) && mxIsChar (pargin [2]))
    {

        //----------------------------------------------------------------------
        // Cout = gbselect (Cin, accum, op, A, desc)
        // Cout = gbselect (Cin, accum, op, A, thunk, desc)
        //----------------------------------------------------------------------

        gb_usage (nargin == 5 || nargin == 6,
            "usage: Cout = gb.select (Cin, accum, op, A, thunk, desc)") ;

        C = gb_get_deep (pargin [0]) ;
// printf ("here %d\n", __LINE__) ;
// GxB_print (C, 2) ;
        OK (GxB_Matrix_type (&ctype, C)) ;
        accum = gb_mxstring_to_binop (pargin [1], ctype) ;
        op = gb_mxstring_to_selectop (pargin [2]) ;
        A = gb_get_shallow (pargin [3]) ;
        thunk = (nargin > 5) ? (GxB_Scalar) gb_get_shallow (pargin [4]) : NULL ;

    }
    else if (mxIsChar (pargin [2]) && !mxIsChar (pargin [3]))
    {

        //----------------------------------------------------------------------
        // Cout = gbselect (Cin, M, op, A, desc)
        // Cout = gbselect (Cin, M, op, A, thunk, desc)
        //----------------------------------------------------------------------

        gb_usage (nargin == 5 || nargin == 6,
            "usage: Cout = gb.select (Cin, M, op, A, thunk, desc)") ;

        C = gb_get_deep (pargin [0]) ;
        M = gb_get_shallow (pargin [1]) ;
        op = gb_mxstring_to_selectop (pargin [2]) ;
        A = gb_get_shallow (pargin [3]) ;
        thunk = (nargin > 5) ? (GxB_Scalar) gb_get_shallow (pargin [4]) : NULL ;

    }
    else if (mxIsChar (pargin [2]) && mxIsChar (pargin [3]))
    {

        //----------------------------------------------------------------------
        // Cout = gbselect (Cin, M, accum, op, A, desc)
        // Cout = gbselect (Cin, M, accum, op, A, thunk, desc)
        //----------------------------------------------------------------------

        gb_usage (nargin == 6 || nargin == 7,
            "usage: Cout = gb.select (Cin, M, accum, op, A, thunk, desc)") ;

        C = gb_get_deep (pargin [0]) ;
// printf ("here %d\n", __LINE__) ;
// GxB_print (C, 2) ;
        OK (GxB_Matrix_type (&ctype, C)) ;
        M = gb_get_shallow (pargin [1]) ;
        accum = gb_mxstring_to_binop (pargin [2], ctype) ;
        op = gb_mxstring_to_selectop (pargin [3]) ;
        A = gb_get_shallow (pargin [4]) ;
        thunk = (nargin > 6) ? (GxB_Scalar) gb_get_shallow (pargin [5]) : NULL ;

    }
    else
    {
        USAGE ("Cout = gb.select (Cin, M, accum, op, A, thunk, desc)") ;
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
        GrB_Index cnrows = (A_transpose) ? ancols : anrows ;
        GrB_Index cncols = (A_transpose) ? anrows : ancols ;

        // determine the type of C
        if (accum != NULL)
        {
            // if accum is present, use its ztype to determine the type of C
            OK (GxB_BinaryOp_ztype (&ctype, accum)) ;
        }
        else
        {
            // otherwise, C has the same type as A
// printf ("here %d\n", __LINE__) ;
// GxB_print (A, 2) ;
            OK (GxB_Matrix_type (&ctype, A)) ;
        }

        OK (GrB_Matrix_new (&C, ctype, cnrows, cncols)) ;
        fmt = gb_get_format (cnrows, cncols, A, NULL, fmt) ;
        OK (GxB_set (C, GxB_FORMAT, fmt)) ;
    }

    //--------------------------------------------------------------------------
    // handle the NaN case
    //--------------------------------------------------------------------------

    GrB_BinaryOp nan_test = NULL ;
    GrB_Matrix thnk = thunk ;

    if (thunk != NULL)
    {
        // check if thunk is NaN
        GrB_Type thunk_type ;
// printf ("here %d\n", __LINE__) ;
// GxB_print (thunk, 2) ;
        OK (GxB_Matrix_type (&thunk_type, thunk)) ;
        bool thunk_is_nan = false ;
        if (thunk_type == GrB_FP32)
        {
            float thunk_value = 0 ;
            OK (GrB_Matrix_extractElement (&thunk_value, thunk, 0, 0)) ;
            thunk_is_nan = isnan (thunk_value) ;
        }
        else if (thunk_type == GrB_FP64)
        {
            double thunk_value = 0 ;
            OK (GrB_Matrix_extractElement (&thunk_value, thunk, 0, 0)) ;
            thunk_is_nan = isnan (thunk_value) ;
        }

        if (thunk_is_nan)
        {
            // thunk is NaN; create a new nan_test operator if it should be used
            // instead of the built-in GxB_EQ_THUNK or GxB_NE_THUNK operators.
            // These operators do not need a thunk input, since it is now known
            // to be a NaN.
            GrB_Type atype ;
// printf ("here %d\n", __LINE__) ;
// GxB_print (A, 2) ;
            OK (GxB_Matrix_type (&atype, A)) ;
            if (op == GxB_EQ_THUNK && atype == GrB_FP32)
            {
                OK (GxB_SelectOp_new (&nan_test, gb_isnan32, GrB_FP32, NULL)) ;
            }
            else if (op == GxB_EQ_THUNK && atype == GrB_FP64)
            {
                OK (GxB_SelectOp_new (&nan_test, gb_isnan64, GrB_FP64, NULL)) ;
            }
            else if (op == GxB_NE_THUNK && atype == GrB_FP32)
            {
                OK (GxB_SelectOp_new (&nan_test, gb_isnotnan32, GrB_FP32,
                    NULL)) ;
            }
            else if (op == GxB_NE_THUNK && atype == GrB_FP64)
            {
                OK (GxB_SelectOp_new (&nan_test, gb_isnotnan64, GrB_FP64,
                    NULL)) ;
            }
        }

        if (nan_test != NULL)
        {
            // use the new operator instead of the built-in one
            op = nan_test ;
            thnk = NULL ;
        }
    }

    //--------------------------------------------------------------------------
    // compute C<M> += select (A, thnk)
    //--------------------------------------------------------------------------

    OK (GxB_select (C, M, accum, op, A, thnk, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_free (&M)) ;
    OK (GrB_free (&A)) ;
    OK (GrB_free (&thunk)) ;
    OK (GrB_free (&desc)) ;
    OK (GrB_free (&nan_test)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
}

