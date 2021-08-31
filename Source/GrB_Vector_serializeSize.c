//------------------------------------------------------------------------------
// GrB_Vector_serializeSize: return an upper bound on the blob size
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_Vector_serialize and GxB_Vector_serialize both serialize a GrB_Vector
// into a blob of bytes.  This function provides an estimate of the # of bytes
// the blob would have, assuming the default method and default # of threads,
// using the dryrun option in GB_serialize.

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GrB_Vector_serializeSize   // estimate the size of a blob
(
    // output:
    size_t *blob_size_handle,       // upper bound on the required size of the
                                    // blob on output.
    // input:
    GrB_Vector u                    // vector to serialize
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Vector_serializeSize (&blob_size, u)") ;
    GB_BURBLE_START ("GrB_Vector_serialize") ;
    GB_RETURN_IF_NULL (blob_size_handle) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;

    // no descriptor, so assume the default method
    int method = GxB_DEFAULT ;

    // Context will hold the default # of threads, which can be controlled
    // by GxB_Global_Option_set.

    //--------------------------------------------------------------------------
    // serialize the vector
    //--------------------------------------------------------------------------

    GrB_Info info = GB_serialize (NULL, blob_size_handle, (GrB_Matrix) u,
        method, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

