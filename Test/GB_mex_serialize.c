//------------------------------------------------------------------------------
// GB_mex_serialize: copy a matrix, using serialize/deserialize
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// copy a matrix using GxB_Matrix_serialize and GxB_Matrix_deserialize

#include "GB_mex.h"

// method:
// GxB_COMPRESSION_NONE -1     // no compression
// GxB_COMPRESSION_DEFAULT 0   // LZ4
// GxB_COMPRESSION_LZ4   1000  // LZ4
// GxB_COMPRESSION_LZ4HC 2000  // LZ4HC, with default level 9
// GxB_COMPRESSION_LZ4HC 2001  // LZ4HC:1
// ...
// GxB_COMPRESSION_LZ4HC 2009  // LZ4HC:9

#define USAGE "C = GB_mex_serialize (A, method, mode)"

#define FREE_ALL                        \
{                                       \
    mxFree (blob) ;                     \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&C) ;              \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, C = NULL ;
    GrB_Descriptor desc = NULL ;
    void *blob = NULL ;
    size_t blob_size = 0 ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY  ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get the type of A
    GrB_Type atype ;
    GxB_Matrix_type (&atype, A) ;

    // get method
    int GET_SCALAR (1, int, method, 0) ;
    if (method != 0)
    {
        GrB_Descriptor_new (&desc) ;
        GxB_Desc_set (desc, GxB_COMPRESSION, method) ;
    }

    // get mode: 0:NULL, 1:fast, 502:secure
    int GET_SCALAR (2, int, mode, GxB_DEFAULT) ;
    if (mode != GxB_DEFAULT)
    {
        if (mode != GxB_SECURE_IMPORT) mode = GxB_FAST_IMPORT ;
        if (desc == NULL) GrB_Descriptor_new (&desc) ;
        GrB_Descriptor_set (desc, GxB_IMPORT, mode) ;
    }

    // serialize A into the blob and then deserialize into C
    if (GB_VECTOR_OK (A))
    {
        // test the vector methods
        METHOD (GxB_Vector_serialize (&blob, &blob_size, (GrB_Vector) A, desc)) ;
        METHOD (GxB_Vector_deserialize ((GrB_Vector *) &C, blob, blob_size, atype, desc)) ;
    }
    else
    {
        // test the matrix methods
        METHOD (GxB_Matrix_serialize (&blob, &blob_size, A, desc)) ;
        METHOD (GxB_Matrix_deserialize (&C, blob, blob_size, atype, desc)) ;
    }

/*
    size_t asize, csize ;
    GxB_Matrix_memoryUsage (&asize, A) ;
    int64_t nallocs ;
    size_t adeep, ashallow ;
    GB_memoryUsage (&nallocs, &adeep, &ashallow, A) ;
    GxB_Matrix_memoryUsage (&csize, C) ;
    printf ("A memory usage:    %ld (shallow %ld, deep %ld, tot %ld)\n", asize,
        adeep, ashallow, adeep+ashallow) ;
    printf ("C memory usage:    %ld\n", csize) ;
    printf ("blob memory usage: %ld (%8.2f%%)\n", blob_size,
        100 * (double) blob_size / (double) csize) ;
*/

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

