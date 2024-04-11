//------------------------------------------------------------------------------
// GB_cuda_apply.hpp: CPU definitions for CUDA apply operations
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_EWISE_H
#define GB_CUDA_EWISE_H

#include "GB_cuda.hpp"

GrB_Info GB_cuda_apply_unop_jit
(
    // output:
    GrB_Matrix C,
    // input:
    GrB_Matrix D,
    GrB_Matrix B,
    GrB_BinaryOp binaryop,
    bool flipxy,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
) ;

GrB_Info GB_cuda_apply_bind1st_jit
(
    // output:
    GrB_Matrix C,
    // input:
    GrB_Matrix A,
    GrB_Matrix D,
    GrB_BinaryOp binaryop,
    bool flipxy,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
) ;

GrB_Info GB_cuda_apply_bind2nd_jit
(
    // output:
    GrB_Matrix C,
    // input:
    GrB_Matrix A,
    GrB_Matrix D,
    GrB_BinaryOp binaryop,
    bool flipxy,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
) ;

#endif

