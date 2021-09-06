//------------------------------------------------------------------------------
// GrB_Vector_serialize: copy a vector into a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// serialize a GrB_Vector into a blob of bytes

// This method is similar to GxB_Vector_serialize.  In contrast with the GrB*
// method, this method requires the user application to allocate the blob
// first, which must be non-NULL on input.  The required size of the blob is
// computed by GrB_Vector_serializeSize.  Example usage:

/*
    void *blob = NULL ;
    size_t blob_size = 0 ;
    GrB_Vector u, w = NULL ;
    // construct a vector u, then serialized it:
    GrB_Vector_serializeSize (&blob_size, u) ;      // loose upper bound
    blob = malloc (blob_size) ;                     // user mallocs the blob
    GrB_Vector_serialize (blob, &blob_size, u) ;    // returns actual size
    blob = realloc (blob, blob_size) ;              // user can shrink the blob
    GrB_Vector_deserialize (&w, blob, blob_size, atype) ;
    free (blob) ;                                   // user frees the blob
*/

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GrB_Vector_serialize       // serialize a GrB_Vector to a blob
(
    // output:
    void *blob,                     // the blob, already allocated in input
    // input/output:
    size_t *blob_size_handle,       // size of the blob on input.  On output,
                                    // the # of bytes used in the blob.
    // input:
    GrB_Vector u                    // vector to serialize
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Vector_serialize (blob, &blob_size, u)") ;
    GB_BURBLE_START ("GrB_Vector_serialize") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (blob_size_handle) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;

    // no descriptor, so assume the default method
    int method = GxB_DEFAULT ;

    // Context will hold the default # of threads, which can be controlled
    // by GxB_Global_Option_set.

    //--------------------------------------------------------------------------
    // serialize the vector into the preallocated blob
    //--------------------------------------------------------------------------

    GrB_Info info = GB_serialize (&blob, blob_size_handle, (GrB_Matrix) u,
        method, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

