//------------------------------------------------------------------------------
// GxB_Matrix_serialize: copy a matrix into a serialized array of bytes
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// serialize a GrB_Matrix into a blob of bytes

#include "GB.h"
#include "GB_serialize.h"

GrB_Info GxB_Matrix_serialize       // serialize a GrB_Matrix to a blob
(
    // output:
    void **blob_handle,             // the blob, allocated on output
    size_t *blob_size_handle,       // size of the blob
    // input:
    const GrB_Matrix A,             // matrix to serialize
    const GrB_Descriptor desc       // descriptor to select compression method
                                    // and to control # of threads used
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_serialize (&blob, &blob_size, A, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_serialize") ;
    GB_RETURN_IF_NULL (blob_handle) ;
    GB_RETURN_IF_NULL (blob_size_handle) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    // get the method from the descriptor
    int method = (desc == NULL) ? GxB_DEFAULT : desc->compression ;

//  printf ("\nGxB_Matrix_serialize method: %d\n", method) ;

    //--------------------------------------------------------------------------
    // serialize the matrix
    //--------------------------------------------------------------------------

    GrB_Info info = GB_serialize (blob_handle, blob_size_handle, A, method,
        Context) ;
    GB_BURBLE_END ;
    return (info) ;
}

