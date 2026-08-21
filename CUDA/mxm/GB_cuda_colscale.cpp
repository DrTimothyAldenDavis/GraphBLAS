//------------------------------------------------------------------------------
// GB_cuda_colscale: C = A*D
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "mxm/GB_cuda_ewise.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

GrB_Info GB_cuda_colscale
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    GrB_Info info ;
    int device = 0 ;    // fixme
    cudaStream_t stream = nullptr ;
    GB_OK (GB_cuda_stream_pool_acquire (&stream)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    // determine the geometry of the CUDA kernel launches
    int32_t number_of_sms = GB_Global_gpu_sm_get (device) ;
    int64_t raw_gridsz = GB_ICEIL (anz, GB_CUDA_SCALE_CHUNKSIZE_LOG2) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

    GB_OK (GB_cuda_colscale_jit (C, A, D, 
        semiring->multiply, flipxy, stream, gridsz)) ;
    
    GB_OK (GB_cuda_stream_pool_release (&stream)) ;
    return GrB_SUCCESS ; 
}

