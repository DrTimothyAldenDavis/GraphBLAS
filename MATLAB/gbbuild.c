//------------------------------------------------------------------------------
// gbbuild: build a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Usage:

#include "gbmex.h"

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

    gb_usage (nargin >= 3 && nargin <= 7 && nargin != 4 && nargout <= 1,
        "usage: A = gbbuild (I, J, X, m, n, dup, type)") ;

    //--------------------------------------------------------------------------
    // get I and J
    //--------------------------------------------------------------------------

    GrB_Index ni, Icolon [3], *I = NULL, I_is_list, I_is_allocated ;
    GrB_Index nj, Jcolon [3], *J = NULL, J_is_list, J_is_allocated ;

    gb_mxarray_to_indices (&I, pargin [0], &ni, Icolon, &I_is_list,
        &I_is_allocated) ;

    gb_mxarray_to_indices (&J, pargin [0], &nj, Jcolon, &J_is_list,
        &J_is_allocated) ;

    CHECK_ERROR (!I_is_list, "I must be a list of indices") ;
    CHECK_ERROR (!J_is_list, "J must be a list of indices") ;
    CHECK_ERROR (ni != nj, "I, J, and X must be the same size") ;

    //--------------------------------------------------------------------------
    // get X
    //--------------------------------------------------------------------------

    GrB_Type xtype = gb_mxarray_type (pargin [2]) ;

    GrB_Index nx = mxGetNumberOfElements (pargin [2]) ;

    CHECK_ERROR (ni != nx, "I, J, and X must be the same size") ;
    CHECK_ERROR (!(mxIsNumeric (pargin [2]) || mxIsLogical (pargin [2])),
        "X must be a numeric or logical array") ;
    CHECK_ERROR (mxIsSparse (pargin [2]), "X cannot be sparse") ;

    // void *X = mxGetData (pargin [2]) ;

    //--------------------------------------------------------------------------
    // get m and n if present
    //--------------------------------------------------------------------------

    GrB_Index nrows = 0, ncols = 0 ;

    if (nargin < 4)
    {
        // nrows = max entry in I + 1
        for (int64_t k = 0 ; i < (int64_t) ni ; k++)
        {
            nrows = MAX (nrows, I [k]) ;
        }
        if (ni > 0) nrows++ ;
    }
    else
    {
        // m is provided on input
        CHECK_ERROR (!IS_SCALAR (pargin [3]), "m must be a scalar") ;
        nrows = (GrB_Index) mxGetScalar (pargin [3]) ;
    }

    if (nargin < 5)
    {
        // ncols = max entry in J
        for (int64_t k = 0 ; i < (int64_t) ni ; k++)
        {
            ncols = MAX (ncols, I [k]) ;
        }
        if (ni > 0) ncols++ ;
    }
    else
    {
        // n is provided on input
        CHECK_ERROR (!IS_SCALAR (pargin [4]), "n must be a scalar") ;
        nrows = (GrB_Index) mxGetScalar (pargin [n]) ;
    }

    //--------------------------------------------------------------------------
    // get the dup operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp dup = NULL ;

    if (nargin > 5)
    {
        dup = gb_mxstring_to_binop (pargin [5], xtype) ;
    }

    //--------------------------------------------------------------------------

    // GxB_print (dup, 1) ;

    pargout [0] = mxCreateDoubleScalar (0) ;
}

