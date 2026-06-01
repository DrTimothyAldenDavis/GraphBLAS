//------------------------------------------------------------------------------
// gbformat: get/set the matrix format to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage

// fmt = gbformat ;                   get the global default format (row/col)
// fmt = gbformat (fmt) ;             set the global default format
// [f,sparsity,iso] = gbformat (G) ;  get the format, sparsity, and iso status
//                                    of a matrix (either @GrB or built-in)

// Calls to GrB_* and mx* methods are intermingled since none of the GrB
// methods allocate any memory.

#include "gb_interface.h"

#define USAGE "usage: [f,s,iso] = GrB.format(G), f = GrB.format (f), or f = GrB.format"

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

    GBMX_USAGE (nargin <= 1 && nargout <= 3, USAGE) ;

    //--------------------------------------------------------------------------
    // get/set the format
    //--------------------------------------------------------------------------

    int fmt = GxB_BY_COL ;
    int sparsity = GxB_AUTO_SPARSITY ;
    int iso = false ;

    if (nargin == 0)
    { 

        //----------------------------------------------------------------------
        // format = GrB.format
        //----------------------------------------------------------------------

        // get the global format
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, &fmt, GxB_FORMAT)) ;

    }
    else // if (nargin == 1)
    {

        if (mxIsChar (pargin [0]))
        { 


            //------------------------------------------------------------------
            // GrB.format (format)
            //------------------------------------------------------------------

            // parse the format string
            int ignore ;
            char format_string [LEN+2] ;
            gbmx_mxstring_to_string (format_string, LEN, pargin [0], "format") ;
            bool ok = gb_string_to_format (format_string, &fmt, &ignore) ;
            CHECK_ERROR (!ok, "invalid format") ;
            // set the global format
            OK (GrB_Global_set_INT32 (GrB_GLOBAL, fmt, GxB_FORMAT)) ;

        }
        else if (mxIsClass (pargin [0], "GrB"))
        { 

            //------------------------------------------------------------------
            // GrB.format (G) for a GraphBLAS matrix G
            //------------------------------------------------------------------

            GrB_Matrix A = gbmx_get_grb_matrix (pargin [0]) ;
            CHECK_ERROR (A == NULL, "invalid @GrB matrix") ;
            OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;
            OK (GrB_Matrix_get_INT32 (A, &sparsity, GxB_SPARSITY_STATUS)) ;
            OK (GrB_Matrix_get_INT32 (A, &iso, GxB_ISO)) ;

        }
        else
        { 

            //------------------------------------------------------------------
            // GrB.format (A) for a built-in matrix A
            //------------------------------------------------------------------

            // built-in matrices are always stored by column
            fmt = GxB_BY_COL ;
            // built-in matrices are sparse or full, never hypersparse or bitmap
            sparsity = mxIsSparse (pargin [0]) ? GxB_SPARSE : GxB_FULL ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    pargout [0] = mxCreateString ((fmt == GxB_BY_ROW) ? "by row" : "by col") ;
    if (nargout > 1)
    { 
        char *s ;
        switch (sparsity)
        {
            case GxB_HYPERSPARSE : s = "hypersparse" ; break ;
            case GxB_SPARSE :      s = "sparse"      ; break ;
            case GxB_BITMAP :      s = "bitmap"      ; break ;
            case GxB_FULL :        s = "full"        ; break ;
            default :              s = ""            ; break ;
        }
        pargout [1] = mxCreateString (s) ;
    }
    if (nargout > 2)
    { 
        pargout [2] = mxCreateString (iso ? "iso-valued" : "non-iso-valued") ;
    }

    gb_wrapup ( ) ;
}

