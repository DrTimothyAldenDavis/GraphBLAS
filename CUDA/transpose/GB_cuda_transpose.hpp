//------------------------------------------------------------------------------
// GB_cuda_transpose.hpp: CPU definitions for CUDA transpose operations
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_APPLY_H
#define GB_CUDA_APPLY_H

#include "GB_cuda.hpp"

GrB_Info GB_cuda_transpose_jit
(

    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
) ;

#endif

