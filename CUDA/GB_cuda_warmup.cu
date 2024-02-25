//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_warmup.cu: warmup the GPU
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_warmup (int device)
{
    // allocate 'nothing' just to load the drivers.
    // No need to free the result.
    bool ok = GB_cuda_set_device( device );
    if (!ok)
    {
        printf ("invalid GPU: %d\n", device) ;
        return (false) ;
    }

    double gpu_memory_size = GB_Global_gpu_memorysize_get (device);

    size_t size = 0 ;
    void *p = GB_malloc_memory (1, 1, &size) ;
    if (p == NULL)
    {
        printf ("Hey!! where's da memory???\n") ;
        return (false) ;
    }
    GB_free_memory ( &p, size) ;

    cudaMalloc ( &p, size ) ;
    if (p == NULL)
    {
        printf ("Hey!! where's da GPU???\n") ;
        return (false) ;
    }
    cudaFree (p) ;

    // TODO check for jit cache? or in GB_init?

    return (true) ;
}

