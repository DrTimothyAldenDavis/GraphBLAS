//------------------------------------------------------------------------------
// GxB_Matrix_deserialize: create a matrix from a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Matrix from a blob of bytes

// FIXME: deserialize requires a blob_size input.

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GxB_Matrix_deserialize     // deserialize blob into a GrB_Matrix
(
    // output:
    GrB_Matrix *C,                  // output matrix created from the blob
    // input:
    const void *blob,               // the blob
    size_t blob_size,               // size of the blob
    GrB_Type user_type,             // type of the matrix, if a user-defined
                                    // type.  Ignored if matrix has a built-in
                                    // type.
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_deserialize (&blob, &blob_size, A, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_deserialize") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (C) ;

    // get the descriptor: TODO

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    GrB_Info info = GB_deserialize (C, blob, blob_size, user_type, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

