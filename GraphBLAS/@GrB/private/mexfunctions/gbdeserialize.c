//------------------------------------------------------------------------------
// gbdeserialize: deserialize a blob into a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// gbdeserialize is an interface to GxB_Matrix_deserialize.

// Usage:

// A = gbdeserialize (blob)         % set the type of A from the blob
// A = gbdeserialize (blob, mode)   % mode is 'fast' or 'secure'
// A = gbdeserialize (blob, mode, type)   % for testing only

#include "gb_interface.h"

#define USAGE "usage: A = GrB.deserialize (blob, mode)"

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
        || mxGetN (pargin [0]) != 1, "blob must be uint8 column vector") ;

    //--------------------------------------------------------------------------
    // get the blob and the optional mode and type
    //--------------------------------------------------------------------------

    void *blob = mxGetData (pargin [0]) ;
    size_t blob_size = mxGetM (pargin [0]) ;

    // get the mode: 'fast' or 'secure'
    GrB_Descriptor desc = NULL ;
    if (nargin > 2)
    { 
        GrB_Desc_Value import_mode ;
        #define LEN 256
        char mode_string [LEN+1] ;
        gb_mxstring_to_string (mode_string, LEN, pargin [arg], "mode") ;
        if (MATCH (mode_string, "fast"))
        { 
            import_mode = GxB_FAST_IMPORT ;
        }
        else if (MATCH (mode_string, "secure"))
        { 
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

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    OK (GxB_Matrix_deserialize (&C, blob, blob_size, ctype, desc)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_GRB) ;
    GB_WRAPUP ;
}

