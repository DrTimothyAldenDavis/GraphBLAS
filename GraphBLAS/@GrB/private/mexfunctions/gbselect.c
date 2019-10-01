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

// The 'tril', 'triu', 'diag', 'offdiag', and '*thunk' operators all require
// the thunk scalar.  The thunk must not appear for the '*0' operators.

#include "gb_matlab.h"

#define USAGE "usage: Cout = GrB.select (Cin, M, accum, op, A, thunk, desc)"

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

    gb_usage (nargin >= 3 && nargin <= 7 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    mxArray *Matrix [4], *String [2], *Cell [2] ;
    base_enum_t base ;
    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    int nmatrices, nstrings, ncells ;
    GrB_Descriptor desc ;
    gb_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String, &nstrings,
        Cell, &ncells, &desc, &base, &kind, &fmt) ;

    CHECK_ERROR (nmatrices < 1 || nstrings < 1 || ncells > 0, USAGE) ;

    //--------------------------------------------------------------------------
    // get the select operator
    //--------------------------------------------------------------------------

    GxB_SelectOp op = gb_mxstring_to_selectop (String [nstrings-1]) ;
    bool thunk_required = 
        (op == GxB_TRIL) || (op == GxB_TRIU) ||
        (op == GxB_DIAG) || (op == GxB_OFFDIAG) ||
        (op == GxB_NE_THUNK) || (op == GxB_EQ_THUNK) ||
        (op == GxB_GT_THUNK) || (op == GxB_GE_THUNK) ||
        (op == GxB_LT_THUNK) || (op == GxB_LE_THUNK) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Type atype, ctype = NULL ;
    GrB_Matrix C = NULL, M = NULL, A, thunk = NULL ;

    if (thunk_required)
    {
        if (nmatrices == 1)
        { 
            ERROR ("select operator input is missing") ;
        }
        else if (nmatrices == 2)
        { 
            A     = gb_get_shallow (Matrix [0]) ;
            thunk = gb_get_shallow (Matrix [1]) ;
        }
        else if (nmatrices == 3)
        { 
            C     = gb_get_deep    (Matrix [0]) ;
            A     = gb_get_shallow (Matrix [1]) ;
            thunk = gb_get_shallow (Matrix [2]) ;
        }
        else // if (nmatrices == 4)
        { 
            C     = gb_get_deep    (Matrix [0]) ;
            M     = gb_get_shallow (Matrix [1]) ;
            A     = gb_get_shallow (Matrix [2]) ;
            thunk = gb_get_shallow (Matrix [3]) ;
        }
    }
    else
    {
        if (nmatrices == 1)
        { 
            A     = gb_get_shallow (Matrix [0]) ;
        }
        else if (nmatrices == 2)
        { 
            C     = gb_get_deep    (Matrix [0]) ;
            A     = gb_get_shallow (Matrix [1]) ;
        }
        else if (nmatrices == 3)
        { 
            C     = gb_get_deep    (Matrix [0]) ;
            M     = gb_get_shallow (Matrix [1]) ;
            A     = gb_get_shallow (Matrix [2]) ;
        }
        else // if (nmatrices == 4)
        { 
            ERROR (USAGE) ;
        }
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the accum operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;

    if (nstrings > 1)
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        accum = gb_mxstring_to_binop (String [0], ctype) ;
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

        // C has the same type as A
        OK (GxB_Matrix_type (&ctype, A)) ;

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
    GB_WRAPUP ;
}

