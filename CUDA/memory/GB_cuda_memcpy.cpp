
//------------------------------------------------------------------------------
// GB_cuda_memcpy: copy memory from CPU to GPU, or within GPU(s)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The source arena can be any valid arena: 0 to 7 (GxB_NARENAS-1) denotes a
// CPU arena, while arenas 8 and higher denote a GPU arena.  The destination
// arena is always a GPU (arenas 8 and higher).

#include "memory/GB_cuda_memory.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

GrB_Info GB_cuda_memcpy
(
    void *dest,                 // destination
    const int dest_arena,       // destination arena, or -1 if not known
    const void *src,            // source
    const int src_arena,        // src arena, or -1 if not known
    size_t n,                   // # of bytes to copy
    const GB_Operator op        // GrB_IDENTITY_UINT8
)
{

    GrB_Info info ;
    int dest_device = dest_arena - GxB_NARENAS ;
    int src_device  = src_arena  - GxB_NARENAS ;
    cudaStream_t stream = nullptr ;

    if (n == 0) return (GrB_SUCCESS) ;

    if (src_device >= 0)
    {

        //----------------------------------------------------------------------
        // both src and dest are on a GPU (perhaps different ones)
        //----------------------------------------------------------------------

        GBURBLE ("(cudaMemcpy: %.3f MB, dest device: %d, source device: %d) ",
            ((double) n) / (1024*1024), dest_device, src_device) ;

        cudaError_t result = cudaMemcpy (dest, src, n, cudaMemcpyDefault) ;
        info = (result == cudaSuccess) ? GrB_SUCCESS : GxB_GPU_ERROR ;
        GB_OK (info) ;
    }
    else
    {

        //----------------------------------------------------------------------
        // src is on the CPU, dest is on a GPU: use the JIT
        //----------------------------------------------------------------------

        GBURBLE ("(cuda memcpy: %.3f MB, dest device: %d, source arena: %d) ",
            ((double) n) / (1024*1024), dest_device, src_arena) ;

        GB_OK (GB_cuda_stream_pool_acquire (dest_device, &stream)) ;

        int32_t number_of_sms = GB_Global_gpu_sm_get (dest_device) ;
        int64_t raw_gridsz = GB_ICEIL (n, GB_CUDA_MEMCPY_CHUNKSIZE) ;
        // cap #of blocks to 256 * #of sms
        int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256));
        gridsz = std::max (gridsz, 1) ;

        GB_OK (GB_cuda_memcpy_jit ((GB_void *) dest, (GB_void *) src, n, op,
            dest_device, stream, gridsz)) ;
        GB_OK (GB_cuda_stream_pool_release (&stream)) ;
    }

    GB_FREE_ALL ;
    return GrB_SUCCESS ;
}

