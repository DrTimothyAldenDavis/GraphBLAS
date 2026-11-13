
//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_memcpy.hpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_MEMCPY_H
#define GB_CUDA_MEMCPY_H

#include "GB_cuda.hpp"

GrB_Info GB_cuda_memcpy_jit
(
    GB_void *dest,              // destination
    const GB_void *src,         // source
    size_t n,                   // # of bytes to copy
    const GB_Operator op,       // GrB_IDENTITY_UINT8
    // CUDA stream and launch parameters:
    int device,
    cudaStream_t stream,
    int32_t gridsz
) ;

#endif

