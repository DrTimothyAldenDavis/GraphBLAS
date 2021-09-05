//------------------------------------------------------------------------------
// GxB_Matrix_deserialize: create a matrix from a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// deserialize: create a GrB_Matrix from a blob of bytes

// Identical to GrB_Matrix_deserialize, except that this method has
// a descriptor as the last parameter, to control the # of threads used.

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GxB_Matrix_deserialize     // deserialize blob into a GrB_Matrix
(
    // output:
    GrB_Matrix *C,      // output matrix created from the blob
    // input:
    const void *blob,   // the blob
    size_t blob_size,   // size of the blob
    GrB_Type type,      // type of the matrix C.  Required if the blob holds a
                        // matrix of user-defined type.  May be NULL if blob
                        // holds a built-in type.  If not NULL and the blob
                        // holds a matrix of a built-in type, then C is
                        // typecasted to this requested type.
    const GrB_Descriptor desc       // to control # of threads used and
                        // whether or not the input blob is trusted.
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_deserialize (&blob, &blob_size, A, type, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_deserialize") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (C) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_GET_DESCRIPTOR_IMPORT (desc, fast_import) ;

    //--------------------------------------------------------------------------
    // deserialize the blob into a matrix
    //--------------------------------------------------------------------------

    info = GB_deserialize (C, blob, blob_size, type, fast_import, Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

