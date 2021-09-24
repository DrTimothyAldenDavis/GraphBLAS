//------------------------------------------------------------------------------
// gbdeserialize: deserialize a blob into a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// gbdeserialize is an interface to GrB_Matrix_deserialize.

// Usage:

// A = gbdeserialize (blob)

#include "gb_interface.h"

#define USAGE "usage: A = GrB.deserialize (blob)"

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

    gb_usage ((nargin >= 1 || nargin <= 3) && nargout <= 1, USAGE) ;
    CHECK_ERROR (mxGetClassID (pargin [0]) != mxUINT8_CLASS
        || mxGetN (pargin [0]) != 1, "blob must be a uint8 column vector") ;

    //--------------------------------------------------------------------------
    // get the blob
    //--------------------------------------------------------------------------

    void *blob = mxGetData (pargin [0]) ;
    size_t blob_size = mxGetM (pargin [0]) ;

#if 0
    // get the mode: 'fast' or 'secure' (or 'debug' for testing only)
    bool debug = false ;
    GrB_Descriptor desc = NULL ;
    if (nargin > 1)
    { 
        GrB_Desc_Value import_mode ;
        #define LEN 256
        char mode_string [LEN+1] ;
        gb_mxstring_to_string (mode_string, LEN, pargin [1], "mode") ;
        if (MATCH (mode_string, "fast"))
        { 
            import_mode = GxB_FAST_IMPORT ;
        }
        else if (MATCH (mode_string, "secure"))
        { 
            import_mode = GxB_SECURE_IMPORT ;
        }
        else if (MATCH (mode_string, "debug"))
        {
            // use GrB_Matrix_deserialize, which does not use the descriptor
            debug = true ;
            import_mode = GxB_SECURE_IMPORT ;
        }
        else
        {
            ERROR ("unknown mode") ;
        }
        OK (GrB_Descriptor_new (&desc)) ;
        OK (GrB_Descriptor_set (desc, GxB_IMPORT, import_mode)) ;
    }

    // get the ctype (testing only; not documented)
    GrB_Type ctype = (nargin > 2) ? gb_mxstring_to_type (pargin [2]) : NULL ;
#endif

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    OK (GrB_Matrix_deserialize (&C, NULL, blob, blob_size)) ;

#if 0
    if (debug)
    {
        // test GrB_Matrix_deserialize (not the default)
        OK (GrB_Matrix_deserialize (&C, ctype, blob, blob_size)) ;
    }
    else
    {
        OK (GxB_Matrix_deserialize (&C, ctype, blob, blob_size, desc)) ;
    }
    OK (GrB_Descriptor_free (&desc)) ;
#endif

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_GRB) ;
    GB_WRAPUP ;
}

