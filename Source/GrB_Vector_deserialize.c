//------------------------------------------------------------------------------
// GrB_Vector_deserialize: create a vector from a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Vector from a blob of bytes

// Identical to GxB_Vector_deserialize, except that this method does not take
// a descriptor as the last parameter.

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GrB_Vector_deserialize     // deserialize blob into a GrB_Vector
(
    // output:
    GrB_Vector *w,      // output vector created from the blob
    // input:
    GrB_Type type,      // type of the vector w.  Required if the blob holds a
                        // vector of user-defined type.  May be NULL if blob
                        // holds a built-in type; otherwise must match the
                        // type of w.
    const void *blob,   // the blob
    size_t blob_size    // size of the blob
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Vector_deserialize (&w, type, blob, blob_size)") ;
    GB_BURBLE_START ("GrB_Vector_deserialize") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (w) ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a vector
    //--------------------------------------------------------------------------

    GrB_Info info = GB_deserialize ((GrB_Matrix *) w, type, blob, blob_size,
        Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

