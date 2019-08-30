//------------------------------------------------------------------------------
// gbnew: create a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// X may be a MATLAB sparse matrix, or a MATLAB struct containing a GraphBLAS
// matrix.  G is returned as a MATLAB struct containing a GraphBLAS matrix.

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

    gb_usage (nargin >= 1 && nargin <= 4 && nargout <= 1,
        "usage: G = gb (m,n,type,format) or G = gb (X,type,format)") ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS matrix
    //--------------------------------------------------------------------------

    GrB_Matrix G ;

    if (nargin == 1)
    {

        //----------------------------------------------------------------------
        // G = gb (X)
        //----------------------------------------------------------------------

        // GraphBLAS copy of X, same type and format as X
        G = gb_get_deep (pargin [0]) ;

    }
    else if (nargin == 2)
    {

        //----------------------------------------------------------------------
        // G = gb (X, type)
        // G = gb (X, format)
        // G = gb (m, n)
        //----------------------------------------------------------------------

        if (mxIsChar (pargin [1]))
        {

            //------------------------------------------------------------------
            // G = gb (X, type)
            // G = gb (X, format)
            //------------------------------------------------------------------

            GrB_Type type = gb_mxstring_to_type (pargin [1]) ;
            GxB_Format_Value format = gb_mxstring_to_format (pargin [1]) ;

            if (type != NULL)
            {

                //--------------------------------------------------------------
                // G = gb (X, type)
                //--------------------------------------------------------------

                if (gb_mxarray_is_empty (pargin [0]))
                {
                    OK (GrB_Matrix_new (&G, type, 0, 0)) ;
                }
                else
                {
                    // get a shallow copy and then typecast it to type.
                    // use the same format as X
                    GrB_Matrix X = gb_get_shallow (pargin [0]) ;
                    OK (GxB_get (X, GxB_FORMAT, &format)) ;
                    G = gb_typecast (type, format, X) ;
                    OK (GrB_free (&X)) ;
                }

            }
            else if (format != GxB_NO_FORMAT)
            {

                //--------------------------------------------------------------
                // G = gb (X, format)
                //--------------------------------------------------------------

                // get a deep copy of X and convert it to the requested format
                G = gb_get_deep (pargin [0]) ;
                OK (GxB_set (G, GxB_FORMAT, format)) ;

            }
            else
            {
                ERROR ("unknown type or format") ;
            }

        }
        else if (gb_mxarray_is_scalar (pargin [0]) &&
                 gb_mxarray_is_scalar (pargin [1]))
        {

            //------------------------------------------------------------------
            // G = gb (m, n)
            //------------------------------------------------------------------

            // m-by-n GraphBLAS double matrix, no entries, default format
            GrB_Index nrows = mxGetScalar (pargin [0]) ;
            GrB_Index ncols = mxGetScalar (pargin [1]) ;
            OK (GrB_Matrix_new (&G, GrB_FP64, nrows, ncols)) ;

            // set to BY_COL if column vector, BY_ROW if row vector,
            // use global default format otherwise
            OK (GxB_set (G, GxB_FORMAT, gb_default_format (nrows, ncols))) ;

        }
        else
        {
            USAGE ("usage: G = gb(m,n), G = gb(X,type), or G = gb(X,format)") ;
        }

    }
    else if (nargin == 3)
    {

        //----------------------------------------------------------------------
        // G = gb (m, n, format)
        // G = gb (m, n, type)
        // G = gb (X, type, format)
        // G = gb (X, format, type)
        //----------------------------------------------------------------------

        if (gb_mxarray_is_scalar (pargin [0]) &&
            gb_mxarray_is_scalar (pargin [1]) && mxIsChar (pargin [2]))
        {

            //------------------------------------------------------------------
            // G = gb (m, n, format)
            // G = gb (m, n, type)
            //------------------------------------------------------------------

            // create an m-by-n matrix with no entries
            GrB_Index nrows = mxGetScalar (pargin [0]) ;
            GrB_Index ncols = mxGetScalar (pargin [1]) ;
            GrB_Type type = gb_mxstring_to_type (pargin [2]) ;
            GxB_Format_Value format = gb_mxstring_to_format (pargin [2]) ;

            if (type != NULL)
            {
                // create an m-by-n matrix of the desired type, no entries,
                // use the default format.
                OK (GrB_Matrix_new (&G, type, nrows, ncols)) ;

                // set to BY_COL if column vector, BY_ROW if row vector,
                // use global default format otherwise
                OK (GxB_set (G, GxB_FORMAT, gb_default_format (nrows, ncols))) ;

            }
            else if (format != GxB_NO_FORMAT)
            {
                // create an m-by-n double matrix of the desired format
                OK (GrB_Matrix_new (&G, GrB_FP64, nrows, ncols)) ;
                OK (GxB_set (G, GxB_FORMAT, format)) ;
            }
            else
            {
                ERROR ("unknown type or format") ;
            }

        }
        else if (mxIsChar (pargin [1]) && mxIsChar (pargin [2]))
        {

            //------------------------------------------------------------------
            // G = gb (X, type, format)
            // G = gb (X, format, type)
            //------------------------------------------------------------------

            GrB_Type type = gb_mxstring_to_type (pargin [1]) ;
            GxB_Format_Value format = gb_mxstring_to_format (pargin [2]) ;

            if (type != NULL && format != GxB_NO_FORMAT)
            {
                // G = gb (X, type, format)
            }
            else
            {
                // G = gb (X, format, type)
                format = gb_mxstring_to_format (pargin [1]) ;
                type = gb_mxstring_to_type (pargin [2]) ;
            }

            if (type == NULL || format == GxB_NO_FORMAT)
            {
                ERROR ("unknown type and/or format") ;
            }

            if (gb_mxarray_is_empty (pargin [0]))
            {
                OK (GrB_Matrix_new (&G, type, 0, 0)) ;
                OK (GxB_set (G, GxB_FORMAT, format)) ;
            }
            else
            {
                // get a shallow copy, typecast it, and set the format
                GrB_Matrix X = gb_get_shallow (pargin [0]) ;
                G = gb_typecast (type, format, X) ;
                OK (GrB_free (&X)) ;
            }
        }
        else
        {
            ERROR ("unknown usage") ;
        }

    }
    else if (nargin == 4)
    {

        //----------------------------------------------------------------------
        // G = gb (m, n, type, format)
        // G = gb (m, n, format, type)
        //----------------------------------------------------------------------

        if (gb_mxarray_is_scalar (pargin [0]) &&
            gb_mxarray_is_scalar (pargin [1]) &&
            mxIsChar (pargin [2]) && mxIsChar (pargin [3]))
        {

            // create an m-by-n matrix with no entries, of the requested
            // type and format
            GrB_Index nrows = mxGetScalar (pargin [0]) ;
            GrB_Index ncols = mxGetScalar (pargin [1]) ;

            GrB_Type type = gb_mxstring_to_type (pargin [2]) ;
            GxB_Format_Value format = gb_mxstring_to_format (pargin [3]) ;

            if (type != NULL && format != GxB_NO_FORMAT)
            {
                // G = gb (m, n, type, format)
            }
            else
            {
                // G = gb (m, n, format, type)
                format = gb_mxstring_to_format (pargin [2]) ;
                type = gb_mxstring_to_type (pargin [3]) ;
            }

            if (type == NULL || format == GxB_NO_FORMAT)
            {
                ERROR ("unknown type and/or format") ;
            }

            OK (GrB_Matrix_new (&G, type, nrows, ncols)) ;
            OK (GxB_set (G, GxB_FORMAT, format)) ;

        }
        else
        {
            ERROR ("unknown usage") ;
        }

    }
    else
    {
        USAGE ("unknown usage") ;
    }

    //--------------------------------------------------------------------------
    // export the output matrix A back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&G, KIND_GB) ;
}

