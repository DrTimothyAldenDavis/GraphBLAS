//------------------------------------------------------------------------------
// GrB_Matrix_deserialize: create a matrix from a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Matrix from a blob of bytes

// Identical to GxB_Matrix_deserialize, except that this method does not take
// a descriptor as the last parameter.  Since it has no descriptor, this method
// cannot be told to trust the input blob, and thus fast_import is false.
// This method is thus slower than GxB_Matrix_deserialize when the input blob
// can be trusted.  If the input data comes from a trusted source, then use
// GxB_Matrix_deserialize, which assumes (by default) that the input data is
// trusted.

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GrB_Matrix_deserialize     // deserialize blob into a GrB_Matrix
(
    // output:
    GrB_Matrix *C,      // output matrix created from the blob
    // input:
    const void *blob,   // the blob
    size_t blob_size,   // size of the blob
    GrB_Type type       // type of the matrix C.  Required if the blob holds a
                        // matrix of user-defined type.  May be NULL if blob
                        // holds a built-in type.  If not NULL and the blob
                        // holds a matrix of a built-in type, then C is
                        // typecasted to this requested type.
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_deserialize (&blob, &blob_size, A, type, desc)") ;
    GB_BURBLE_START ("GrB_Matrix_deserialize") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (C) ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    bool fast_import = false ;
    GrB_Info info = GB_deserialize (C, blob, blob_size, type, fast_import,
        Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

