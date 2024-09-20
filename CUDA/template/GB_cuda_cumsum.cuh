//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_cumsum: cumlative sum of an array on the GPU(s)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_CUMSUM
#define GB_CUDA_CUMSUM

#include <cub/cub.h>

typedef enum GB_CUDA_CUMSUM_TYPE
{
    GB_CUDA_CUMSUM_EXCLUSIVE_IN_PLACE,
    GB_CUDA_CUMSUM_INCLUSIVE_IN_PLACE,
    GB_CUDA_CUMSUM_EXCLUSIVE,
    GB_CUDA_CUMSUM_INCLUSIVE
} GB_CUDA_CUMSUM_TYPE;

__host__ GrB_Info GB_cuda_cumsum             // compute the cumulative sum of an array
(
    int64_t *restrict in,    // size n or n+1, input
    int64_t *restrict out,   // ignored if in-place sum, else size n or n+1
    const int64_t n,
    cudaStream_t stream,
    GB_CUDA_CUMSUM_TYPE type
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (in != NULL) ;

    if (type >= 2)
    {
        ASSERT (out != NULL) ;
    } 
    else 
    {
        out = in;
        type = (type == 0 ? GB_CUDA_CUMSUM_INCLUSIVE :
                            GB_CUDA_CUMSUM_EXCLUSIVE) ;
    }

    ASSERT (n >= 0) ;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0 ;

    switch (type)
    {
        case GB_CUDA_CUMSUM_INCLUSIVE:
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n, stream) ;
            break;
        default:
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n, stream) ;
    }

    CU_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run
    switch (type)
    {
        case GB_CUDA_CUMSUM_INCLUSIVE:
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n, stream) ;
            break;
        default:
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n, stream) ;
    }

    CU_TRY(cudaFree(d_temp_storage));

    return GrB_SUCCESS;
}
#endif
