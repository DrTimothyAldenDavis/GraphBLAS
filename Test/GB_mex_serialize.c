//------------------------------------------------------------------------------
// GB_mex_serialize: copy a matrix, using serialize/deserialize
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// copy a matrix using GxB_Matrix_serialize and GxB_Matrix_deserialize

#include "GB_mex.h"

#define USAGE "C = GB_mex_serialize (A, method, sparsity)"

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
    GxB_Matrix_fprint (A, "got A", 2, stdout) ;

    // get the type of A
    GrB_Type atype ;
    GxB_Matrix_type (&atype, A) ;

    // get method
    int GET_SCALAR (1, int, method, 0) ;

    // get sparsity
    int GET_SCALAR (2, int, sparsity, GxB_DEFAULT) ;

    // copy C with the same type as A, with default sparsity
    GxB_Matrix_serialize (&blob, &blob_size, A, NULL) ;
    int info = GxB_Matrix_deserialize (&C, blob, atype, NULL) ;
    printf ("got C: %d\n", info) ;
    GxB_Matrix_fprint (C, "got C", 2, stdout) ;
    printf ("blob %p\n", blob) ;
    GB_AS_IF_FREE (blob) ;
    mxFree (blob) ;

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

    // return C as a struct and free the GraphBLAS C
    printf ("return C\n") ;
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
    printf ("bye C\n") ;

    FREE_ALL ;
}

