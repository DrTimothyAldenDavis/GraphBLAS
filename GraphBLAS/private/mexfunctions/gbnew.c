//------------------------------------------------------------------------------
// gbnew: create a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// X may be a MATLAB sparse matrix, or a MATLAB struct containing a GraphBLAS
// matrix.  A is returned as a MATLAB struct containing a GraphBLAS matrix.

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

    gb_usage (nargin >= 1 && nargin <= 3 && nargout <= 1,
        "usage: G = gb (m,n,type) or G = gb (X,type)") ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS matrix
    //--------------------------------------------------------------------------

    GrB_Matrix G ;

    if (nargin == 1)
    {

        //----------------------------------------------------------------------
        // G = gb (X)
        //----------------------------------------------------------------------

        // GraphBLAS copy of X, same type as X
        G = gb_get_deep (pargin [0], NULL) ;

    }
    else if (nargin == 2)
    {

        //----------------------------------------------------------------------
        // G = gb (X, type)
        // G = gb (m, n)
        //----------------------------------------------------------------------

        if (mxIsChar (pargin [1]))
        {

            //------------------------------------------------------------------
            // G = gb (X, type)
            //------------------------------------------------------------------

            GrB_Type xtype = gb_mxstring_to_type (pargin [1]) ;
            if (gb_mxarray_is_empty (pargin [0]))
            {
                OK (GrB_Matrix_new (&G, xtype, 0, 0)) ;
            }
            else
            {
                G = gb_get_deep (pargin [0], xtype) ;
            }

        }
        else if (gb_mxarray_is_scalar (pargin [0]) &&
                 gb_mxarray_is_scalar (pargin [1]))
        {

            //------------------------------------------------------------------
            // G = gb (m, n)
            //------------------------------------------------------------------

            // m-by-n GraphBLAS double matrix with no entries
            GrB_Index nrows = mxGetScalar (pargin [0]) ;
            GrB_Index ncols = mxGetScalar (pargin [1]) ;
            OK (GrB_Matrix_new (&G, GrB_FP64, nrows, ncols)) ;

        }
        else
        {
            USAGE ("usage: G = gb(m,n), or G = gb(X,type)") ;
        }

    }
    else if (nargin == 3)
    {

        //----------------------------------------------------------------------
        // G = gb (m, n, type)
        //----------------------------------------------------------------------

        if (gb_mxarray_is_scalar (pargin [0]) &&
            gb_mxarray_is_scalar (pargin [1]) && mxIsChar (pargin [2]))
        {

            // create an m-by-n matrix of the desired type, with no entries
            GrB_Index nrows = mxGetScalar (pargin [0]) ;
            GrB_Index ncols = mxGetScalar (pargin [1]) ;
            GrB_Type type = gb_mxstring_to_type (pargin [2]) ;
            OK (GrB_Matrix_new (&G, type, nrows, ncols)) ;

        }
        else
        {
            USAGE ("usage: G = gb (m,n,type)") ;
        }
    }

    //--------------------------------------------------------------------------
    // export the output matrix A back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&G, KIND_GB) ;
}

