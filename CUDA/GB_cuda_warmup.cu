//------------------------------------------------------------------------------
// GB_cuda_warmup.cu: warmup the GPU
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0
// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_cuda.h"
/*
#include "rmm/include/rmm/mr/device/managed_memory_resource.hpp"
#include "rmm/include/rmm/mr/device/pool_memory_resource.hpp"
#include "rmm/include/rmm/mr/device/owning_wrapper.hpp"
#include "rmm/include/rmm/mr/device/default_memory_resource.hpp"
#include "rmm/include/rmm/mr/device/per_device_resource.hpp"
#include "rmm/include/rmm/mr/device/cnmem_managed_memory_resource.hpp"
*/
// #include "rmm/detail/cnmem.h"

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

    printf ("warming up device %d memsize %g sms %d\n",
        device,
        gpu_memory_size, 
        GB_Global_gpu_sm_get (device)) ;

    size_t size = 0 ;
    void *p = GB_malloc_memory (1, 1, &size) ;
    if (p == NULL)
    {
        printf ("Hey!! where's da memory???\n") ;
        return (false) ;
    }
    printf ("oooo nice block of memory of size %lu\n", size) ;
    GB_free_memory (&p, size) ;

    p = cudaMalloc (1) ;
    if (p == NULL)
    {
        printf ("Hey!! where's da GPU???\n") ;
        return (false) ;
    }
    cudaFree (p) ;

    printf ("GPU %d nice and toasty now\n", device) ;

    // TODO check for jit cache? or in GB_init?

    return  true; //(err == cudaSuccess) ;
}

