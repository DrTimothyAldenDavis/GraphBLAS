//------------------------------------------------------------------------------
// gbserialize: serialize a matrix into a blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbserialize is an interface to GxB_Matrix_serialize.

// Usage:

// blob = gbserialize (A, method)

// The blob is returned as the opaque content of an n-by-1 uint8 @GrB matrix.

#define FREE_WORK                   \
    GrB_Matrix_free (&A_shallow) ;  \
    GrB_Descriptor_free (&desc) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    gb_free ((void **) &blob) ;     \
    GrB_Vector_free (&Blob) ;

#include "gb_interface.h"

#define USAGE "usage: blob = GrB.serialize (A, method, level)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix *Blob_opaque = NULL, A = NULL, A_shallow = NULL ;
    GrB_Vector Blob = NULL ;
    GrB_Descriptor desc = NULL ;
    void *blob = NULL ;

    gbmx_usage ((nargin >= 1 && nargin <= 3) && nargout <= 1, USAGE) ;
    pargout [0] = gbmx_export_struct (&Blob_opaque) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    char method_name [LEN+2] ;

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;

    int method = GxB_COMPRESSION_DEFAULT ;
    int level = 0 ;     // use whatever is the default for the method

    if (nargin > 1)
    { 
        gbmx_mxstring_to_string (method_name, LEN, pargin [1], "method") ;
    }

    // get the method level
    if (nargin > 2)
    { 
        level = (int) mxGetScalar (pargin [2]) ;
    }
    if (level < 0 || level > 999) level = 0 ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get input matrix
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_shallow, &(Matrix [0]))) ;

    //--------------------------------------------------------------------------
    // create descriptor
    //--------------------------------------------------------------------------

    bool debug = false ;
    if (nargin > 1)
    { 
        // create the descriptor
        OK (GrB_Descriptor_new (&desc)) ;
        // get the method
        if (MATCH (method_name, "none"))
        { 
            method = GxB_COMPRESSION_NONE ;
        }
        else if (MATCH (method_name, "lz4"))
        { 
            method = GxB_COMPRESSION_LZ4 ;
        }
        else if (MATCH (method_name, "lz4hc"))
        { 
            method = GxB_COMPRESSION_LZ4HC ;
        }
        else if (MATCH (method_name, "default") || MATCH (method_name, "zstd"))
        { 
            // the default is ZSTD, with level 1
            method = GxB_COMPRESSION_ZSTD ;
        }
        else if (MATCH (method_name, "debug"))
        { 
            // use GrB_Matrix_serializeSize and GrB_Matrix_serialize, just
            // for testing
            debug = true ;
        }
        else
        { 
            ERROR ("unknown method", GrB_INVALID_VALUE) ;
        }
        // set the descriptor
        OK (GrB_Descriptor_set_INT32 (desc, method + level, GxB_COMPRESSION)) ;
    }

    //--------------------------------------------------------------------------
    // serialize the matrix into the blob (in arena 0)
    //--------------------------------------------------------------------------

    uint64_t blob_memsize = 0 ;

    if (debug)
    { 
        // debug GrB_Matrix_serializeSize and GrB_Matrix_serialize
        OK (GrB_Matrix_serializeSize (&blob_memsize, A)) ;
        blob = gb_malloc (blob_memsize) ;
        OK (GrB_Matrix_serialize (blob, &blob_memsize, A)) ;
        // shrink the blob to its actual size
        // blob = realloc (blob, blob_memsize) ;    // this is skipped
    }
    else
    { 
        // use GxB_Matrix_serialize by default
        OK (GxB_Matrix_serialize (&blob, &blob_memsize, A, desc)) ;
    }

    //--------------------------------------------------------------------------
    // transfer the blob into the output Blob vector
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&Blob, GrB_UINT8, blob_memsize)) ;
    OK (GxB_Vector_load (Blob, &blob, GrB_UINT8, blob_memsize, blob_memsize,
        GrB_DEFAULT, NULL)) ;
    ASSERT (blob == NULL) ;

    //--------------------------------------------------------------------------
    // free workspace and return results
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (Blob_opaque, (GrB_Matrix *) &Blob, KIND_GRB)) ;
    gb_wrapup ( ) ;
}

