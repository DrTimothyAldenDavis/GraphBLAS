//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_error.h
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

/*
 * Copyright (c) 2023 NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef GB_CUDA_ERROR_H
#define GB_CUDA_ERROR_H

#include <cuda_runtime.h>

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                cudaGetErrorString(err));
    }
}
#define CHECK_CUDA(call) checkCudaErrors( call )

//------------------------------------------------------------------------------
// CUDA_OK: like GB_OK but for calls to cuda* methods
//------------------------------------------------------------------------------

// FIXME: always use this method, not the above methods.

// FIXME: GrB_NO_VALUE means something in CUDA failed, and the caller will then
// do the computation on the CPU.  Need to turn off the JIT for CUDA kernels
// (but not CPU kernels) if some CUDA error occurred.  Current JIT control does
// not distinguish between CPU and CUDA failures.

#define CUDA_OK(cudaMethod)                                                 \
{                                                                           \
    cudaError_t cuda_error = cudaMethod ;                                   \
    if (cuda_error != cudaSuccess)                                          \
    {                                                                       \
        GrB_Info info = (cuda_error == cudaErrorMemoryAllocation) ?         \
            GrB_OUT_OF_MEMORY : GrB_NO_VALUE ;                              \
        GBURBLE ("(cuda failed: %d:%s file:%s line:%d) ", (int) cuda_error, \
            cudaGetErrorString (cuda_error), __FILE__, __LINE__) ;          \
        GB_FREE_ALL ;                                                       \
        return (info) ;                                                     \
    }                                                                       \
}

#endif
