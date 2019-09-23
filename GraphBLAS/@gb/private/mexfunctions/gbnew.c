//------------------------------------------------------------------------------
// gbnew: create a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A may be a MATLAB sparse matrix, or a MATLAB struct containing a GraphBLAS
// matrix.  C is returned as a MATLAB struct containing a GraphBLAS matrix.

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
        "usage: C = gb (m,n,type,format) or C = gb (A,type,format)") ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C ;

    if (nargin == 1)
    { 

        //----------------------------------------------------------------------
        // C = gb (A)
        //----------------------------------------------------------------------

        // GraphBLAS copy of A, same type and format as A
        C = gb_get_deep (pargin [0]) ;

    }
    else if (nargin == 2)
    {

        //----------------------------------------------------------------------
        // C = gb (A, type)
        // C = gb (A, format)
        // C = gb (m, n)
        //----------------------------------------------------------------------

        if (mxIsChar (pargin [1]))
        {

            //------------------------------------------------------------------
            // C = gb (A, type)
            // C = gb (A, format)
            //------------------------------------------------------------------

            GrB_Type type = gb_mxstring_to_type (pargin [1]) ;
            GxB_Format_Value fmt = gb_mxstring_to_format (pargin [1]) ;

            if (type != NULL)
            {

                //--------------------------------------------------------------
                // C = gb (A, type)
                //--------------------------------------------------------------

                if (gb_mxarray_is_empty (pargin [0]))
                { 
                    // TODO get the format of A
                    OK (GrB_Matrix_new (&C, type, 0, 0)) ;
                }
                else
                { 
                    // get a shallow copy and then typecast it to type.
                    // use the same format as A
                    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
                    OK (GxB_get (A, GxB_FORMAT, &fmt)) ;
                    C = gb_typecast (type, fmt, A) ;
                    OK (GrB_free (&A)) ;
                }

            }
            else if (fmt != GxB_NO_FORMAT)
            { 

                //--------------------------------------------------------------
                // C = gb (A, format)
                //--------------------------------------------------------------

                // get a deep copy of A and convert it to the requested format
                C = gb_get_deep (pargin [0]) ;
                OK (GxB_set (C, GxB_FORMAT, fmt)) ;

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
            // C = gb (m, n)
            //------------------------------------------------------------------

            // m-by-n GraphBLAS double matrix, no entries, default format
            GrB_Index nrows = mxGetScalar (pargin [0]) ;
            GrB_Index ncols = mxGetScalar (pargin [1]) ;
            OK (GrB_Matrix_new (&C, GrB_FP64, nrows, ncols)) ;

            // set to BY_COL if column vector, BY_ROW if row vector,
            // use global default format otherwise
            OK (GxB_set (C, GxB_FORMAT, gb_default_format (nrows, ncols))) ;

        }
        else
        { 
            ERROR ("usage: C = gb(m,n), C = gb(A,type), or C = gb(A,format)") ;
        }

    }
    else if (nargin == 3)
    {

        //----------------------------------------------------------------------
        // C = gb (m, n, format)
        // C = gb (m, n, type)
        // C = gb (A, type, format)
        // C = gb (A, format, type)
        //----------------------------------------------------------------------

        if (gb_mxarray_is_scalar (pargin [0]) &&
            gb_mxarray_is_scalar (pargin [1]) && mxIsChar (pargin [2]))
        {

            //------------------------------------------------------------------
            // C = gb (m, n, format)
            // C = gb (m, n, type)
            //------------------------------------------------------------------

            // create an m-by-n matrix with no entries
            GrB_Index nrows = mxGetScalar (pargin [0]) ;
            GrB_Index ncols = mxGetScalar (pargin [1]) ;
            GrB_Type type = gb_mxstring_to_type (pargin [2]) ;
            GxB_Format_Value fmt = gb_mxstring_to_format (pargin [2]) ;

            if (type != NULL)
            { 
                // create an m-by-n matrix of the desired type, no entries,
                // use the default format.
                OK (GrB_Matrix_new (&C, type, nrows, ncols)) ;

                // set to BY_COL if column vector, BY_ROW if row vector,
                // use global default format otherwise
                OK (GxB_set (C, GxB_FORMAT, gb_default_format (nrows, ncols))) ;

            }
            else if (fmt != GxB_NO_FORMAT)
            { 
                // create an m-by-n double matrix of the desired format
                OK (GrB_Matrix_new (&C, GrB_FP64, nrows, ncols)) ;
                OK (GxB_set (C, GxB_FORMAT, fmt)) ;
            }
            else
            { 
                ERROR ("unknown type or format") ;
            }

        }
        else if (mxIsChar (pargin [1]) && mxIsChar (pargin [2]))
        {

            //------------------------------------------------------------------
            // C = gb (A, type, format)
            // C = gb (A, format, type)
            //------------------------------------------------------------------

            GrB_Type type = gb_mxstring_to_type (pargin [1]) ;
            GxB_Format_Value fmt = gb_mxstring_to_format (pargin [2]) ;

            if (type != NULL && fmt != GxB_NO_FORMAT)
            { 
                // C = gb (A, type, format)
            }
            else
            { 
                // C = gb (A, format, type)
                fmt = gb_mxstring_to_format (pargin [1]) ;
                type = gb_mxstring_to_type (pargin [2]) ;
            }

            if (type == NULL || fmt == GxB_NO_FORMAT)
            { 
                ERROR ("unknown type and/or format") ;
            }

            if (gb_mxarray_is_empty (pargin [0]))
            { 
                OK (GrB_Matrix_new (&C, type, 0, 0)) ;
                OK (GxB_set (C, GxB_FORMAT, fmt)) ;
            }
            else
            { 
                // get a shallow copy, typecast it, and set the format
                GrB_Matrix A = gb_get_shallow (pargin [0]) ;
                C = gb_typecast (type, fmt, A) ;
                OK (GrB_free (&A)) ;
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
        // C = gb (m, n, type, format)
        // C = gb (m, n, format, type)
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
            GxB_Format_Value fmt = gb_mxstring_to_format (pargin [3]) ;

            if (type != NULL && fmt != GxB_NO_FORMAT)
            { 
                // C = gb (m, n, type, format)
            }
            else
            { 
                // C = gb (m, n, format, type)
                fmt = gb_mxstring_to_format (pargin [2]) ;
                type = gb_mxstring_to_type (pargin [3]) ;
            }

            if (type == NULL || fmt == GxB_NO_FORMAT)
            { 
                ERROR ("unknown type and/or format") ;
            }

            OK (GrB_Matrix_new (&C, type, nrows, ncols)) ;
            OK (GxB_set (C, GxB_FORMAT, fmt)) ;

        }
        else
        { 
            ERROR ("unknown usage") ;
        }

    }
    else
    { 
        ERROR ("unknown usage") ;
    }

    //--------------------------------------------------------------------------
    // export the output matrix C back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_GB) ;
    GB_WRAPUP ;
}

