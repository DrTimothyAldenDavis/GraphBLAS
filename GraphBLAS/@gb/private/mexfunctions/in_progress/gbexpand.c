//------------------------------------------------------------------------------
// gbexpand: expand a scalar into a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C = gbexpand (scalar, S, desc)
// C = gbexpand (scalar, m, n, desc)

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

    gb_usage ((nargin == 3 || nargin == 4) && nargout <= 1,
        "usage: C = gbexpand (s, S, desc) or C = gbexpand (s, m, n, desc)") ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, scalar = NULL ;
    GrB_Index cnrows, cncols ;
    GrB_Type ctype ;

    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    GrB_Descriptor desc = 
        gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt) ;

    //--------------------------------------------------------------------------
    // get the scalar
    //--------------------------------------------------------------------------

    scalar = gb_get_shallow (pargin [0]) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, scalar)) ;
    OK (GrB_Matrix_nrows (&ncols, scalar)) ;
    if (nrows != 1 || ncols != 1)
    {
        ERROR ("s must be a scalar") ;
    }
    OK (GxB_Matrix_type (&ctype, scalar)) ;

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    if (nargin == 3)
    {

        //----------------------------------------------------------------------
        // C = gbexpand (scalar, S, desc)
        //----------------------------------------------------------------------

        GrB_Matrix S = gb_get_shallow (pargin [1]) ;
        OK (GrB_Matrix_nrows (&cnrows, S)) ;
        OK (GrB_Matrix_nrows (&cncols, S)) ;
        fmt = gb_get_format (cnrows, cncols, S, NULL, fmt) ;
        OK (GrB_Matrix_new (&C, ctype, cnrows, cncols)) ;
        OK (GrB_free (&S)) ;

    }
    else if (nargin == 4)
    {

        //----------------------------------------------------------------------
        // C = gbexpand (scalar, m, n, desc)
        //----------------------------------------------------------------------

        cnrows = (GrB_Index) mxGetScalar (pargin [1]) ;
        cncols = (GrB_Index) mxGetScalar (pargin [2]) ;
        OK (GrB_Matrix_new (&C, ctype, cnrows, cncols)) ;
        fmt = gb_get_format (cnrows, cncols, scalar, NULL, fmt) ;
        OK (GxB_set (C, GxB_FORMAT, fmt)) ;

    }

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_free (&scalar)) ;
    OK (GrB_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
}

