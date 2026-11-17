//------------------------------------------------------------------------------
// GB_cuda_memset: memset on the GPU
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The destination arena is always a GPU (arenas 8 and higher).

#include "memory/GB_cuda_memory.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

GrB_Info GB_cuda_memset
(
    void *dest,                 // destination
    const int dest_arena,       // destination arena, or -1 if not known
    const int c,                // value to set
    size_t n                    // # of bytes to set
)
{
    GrB_Info info ;
    int dest_device = dest_arena - GxB_NARENAS ;
    cudaStream_t stream = nullptr ;

    if (n == 0) return (GrB_SUCCESS) ;

    if (dest_device < 0) return (GrB_INVALID_VALUE) ;

    GB_OK (GB_cuda_stream_pool_acquire (dest_device, &stream)) ;
    CUDA_OK (cudaSetDevice (dest_device)) ;
    CUDA_OK (cudaMemsetAsync (dest, c, n, stream)) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

