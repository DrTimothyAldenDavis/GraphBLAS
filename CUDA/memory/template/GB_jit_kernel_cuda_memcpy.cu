//------------------------------------------------------------------------------
// CUDA/memory/template/GB_jit_kernel_cuda_memcpy
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GB_cuda_memcpy_kernel: device kernel for memcpy
//------------------------------------------------------------------------------

__global__ void GB_cuda_memcpy_kernel
(
    GB_void *dest,
    const GB_void *src,
    size_t n
)
{
    // FIXME: upcast to 32-bit integers and use a cleanup at the end if
    // n is not a multiple of sizeof(uint32_t).  Also try CUB load/store.
    int tid = blockDim.x * blockIdx.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;
    for (int64_t p = tid ; p < n ; p += nthreads)
    {
        dest [p] = src [p] ;
    }
}

//------------------------------------------------------------------------------
// host CUDA JIT kernel for memcpy
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_MEMCPY_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_MEMCPY_PROTO (GB_jit_kernel)
{
    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;
    dim3 block (GB_CUDA_MEMCPY_BLOCKDIM) ;

    if (n == 0) return (GrB_SUCCESS) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_memcpy_kernel <<<grid, block, 0, stream>>> (dest, src, n) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}

