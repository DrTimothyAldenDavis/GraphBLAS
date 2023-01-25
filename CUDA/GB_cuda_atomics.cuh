/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * Specializations for different atomic operations on different types
 */

#pragma once

template <typename T> __device__ void GB_atomic_write (T* ptr, T val) ;
template <typename T> __device__ void GB_atomic_add (T* ptr, T val) ;
template <typename T> __device__ void GB_atomic_times (T* ptr, T val) ;
template <typename T> __device__ void GB_atomic_min (T* ptr, T val) ;
template <typename T> __device__ void GB_atomic_max (T* ptr, T val) ;
template <typename T> __device__ void GB_atomic_bor (T* ptr, T val) ;
template <typename T> __device__ void GB_atomic_band (T* ptr, T val) ;
template <typename T> __device__ void GB_atomic_bxor (T* ptr, T val) ;
template <typename T> __device__ void GB_atomic_bnxor (T* ptr, T val) ;

__device__ __inline__ void GB_cuda_lock   (int32_t *mutex) ;
__device__ __inline__ void GB_cuda_unlock (int32_t *mutex) ;

// reinterpret a value as another type, but with no typecasting
#define pun(type,x) (*(type *) (&(x)))

//------------------------------------------------------------------------------
// GB_atomic_write
//------------------------------------------------------------------------------

// atomic write (16, 32, and 64 bits)
// no atomic write for double complex

template<> __device__ __inline__ void GB_atomic_write<int16_t>(int16_t* ptr, int16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = pun (unsigned short int, val) ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_write<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_write<int32_t>(int32_t* ptr, int32_t val)
{
    // native CUDA method
    atomicExch ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_atomic_write<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // native CUDA method
    atomicExch ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_write<int64_t>(int64_t* ptr, int64_t val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = pun (unsigned long long int, val) ;
    atomicExch (p, v) ;
}

template<> __device__ __inline__ void GB_atomic_write<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // native CUDA method
    atomicExch ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_write<float>(float* ptr, float val)
{
    // native CUDA method
    atomicExch (ptr, val) ;
}

template<> __device__ __inline__ void GB_atomic_write<double>(double* ptr, double val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = pun (unsigned long long int, val) ;
    atomicExch (p, v) ;
}

template<> __device__ __inline__ void GB_atomic_write<float complex>(float complex* ptr, float complex val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = pun (unsigned long long int, val) ;
    atomicExch (p, v) ;
}

//------------------------------------------------------------------------------
// GB_atomic_add for built-in types
//------------------------------------------------------------------------------

// types: int and uint [16,32,64], float, double, float complex, complex double

template<> __device__ __inline__ void GB_atomic_add<int16_t>(int16_t* ptr, int16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value:
        int16_t new_value = pun (int16_t, assumed) + val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned short int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_add<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed + v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_add<int32_t>(int32_t* ptr, int32_t val)
{
    // native CUDA method
    atomicAdd ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_atomic_add<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // native CUDA method
    atomicAdd ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_add<int64_t>(int64_t* ptr, int64_t val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = pun (unsigned long long int, val) ;
    atomicAdd (p, v) ;
}

template<> __device__ __inline__ void GB_atomic_add<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // native CUDA method
    atomicAdd ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_add<float>(float* ptr, float val)
{
    // native CUDA method
    atomicAdd (ptr, val) ;
}

template<> __device__ __inline__ void GB_atomic_add<double>(double* ptr, double val)
{
    // native CUDA method
    atomicAdd (ptr, val) ;
}

template<> __device__ __inline__ void GB_atomic_add<float complex>(float complex* ptr, float  complexval)
{
    // native CUDA method on each float, real and imaginary parts
    float *p = (float *) ptr ;
    atomicAdd (p  , crealf (val)) ;
    atomicAdd (p+1, cimagf (val)) ;
}

template<> __device__ __inline__ void GB_atomic_add<double complex>(double complex* ptr, double complex val)
{
    // native CUDA method on each double, real and imaginary parts
    double *p = (double *) ptr ;
    atomicAdd (p  , creal (val)) ;
    atomicAdd (p+1, cimag (val)) ;
}

//------------------------------------------------------------------------------
// GB_atomic_times for built-in types
//------------------------------------------------------------------------------

// types: int and uint [16,32,64], float, double, float complex
// no double complex.

template<> __device__ __inline__ void GB_atomic_times<int16_t>(int16_t* ptr, int16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value:
        int16_t new_value = pun (int16_t, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned short int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_times<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed * v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_times<int32_t>(int32_t* ptr, int32_t val)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        int32_t new_value = pun (int32_t, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_times<uint32_t>(uint32_t* ptr, uint32_t val)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int v = (unsigned int) val ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed * v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_times<int64_t>(int64_t* ptr, int64_t val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        int64_t new_value = pun (int64_t, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_times<uint64_t>(uint64_t* ptr, uint64_t val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = (unsigned long long int) val ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed * v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_times<float>(float* ptr, float val)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        float new_value = pun (float, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_times<double>(double* ptr, double val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        double new_value = pun (double, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_times<float complex>(float complex* ptr, float complex val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        float complex new_value = pun (float complex, assumed) * val ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

//------------------------------------------------------------------------------
// GB_atomic_min
//------------------------------------------------------------------------------

// types: int and uint [16,32,64], float, and double
// no complex types

template<> __device__ __inline__ void GB_atomic_min<int16_t>(int16_t* ptr, int16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        int16_t assumed_int16 = pun (int16_t, assumed) ;
        int16_t new_value = GB_IMIN (assumed_int16, val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned short int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_min<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        unsigned short int new_value = GB_IMIN (assumed, v) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, new_value) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_min<int32_t>(int32_t* ptr, int32_t val)
{
    // native CUDA method
    atomicMin ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_atomic_min<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // native CUDA method
    atomicMin ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_min<int64_t>(int64_t* ptr, int64_t val)
{
    // native CUDA method
    atomicMin ((long long int *) ptr, (long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_min<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // native CUDA method
    atomicMin ((unsigned long long int *)ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_min<float>(float* ptr, float val)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        float new_value = fminf (pun (float, assumed), val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_min<double>(double* ptr, double val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        double new_value = fmin (pun (double, assumed), val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

//------------------------------------------------------------------------------
// GB_atomic_max
//------------------------------------------------------------------------------

// types: int and uint [16,32,64], float, and double
// no complex types

template<> __device__ __inline__ void GB_atomic_max<int16_t>(int16_t* ptr, int16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        int16_t assumed_int16 = pun (int16_t, assumed) ;
        int16_t new_value = GB_IMIN (assumed_int16, val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned short int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_max<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        unsigned short int new_value = GB_IMIN (assumed, v) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, new_value) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_max<int32_t>(int32_t* ptr, int32_t val)
{
    // native CUDA method
    atomicMin ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_atomic_max<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // native CUDA method
    atomicMin ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_max<int64_t>(int64_t* ptr, int64_t val)
{
    // native CUDA method
    atomicMin ((long long int *) ptr, (long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_max<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // native CUDA method
    atomicMin ((unsigned long long int *)ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_max<float>(float* ptr, float val)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        float new_value = fmaxf (pun (float, assumed), val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned int, new_value)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_max<double>(double* ptr, double val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // compute the new value
        double new_value = fmax (pun (double, assumed), val) ;
        // modify it atomically:
        old = atomicCAS (p, assumed, pun (unsigned long long int, new_value)) ;
    }
    while (assumed != old) ;
}

//------------------------------------------------------------------------------
// GB_atomic_bor
//------------------------------------------------------------------------------

// bitwise: on uint [16,32,64]

template<> __device__ __inline__ void GB_atomic_bor<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed | v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_bor<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // native CUDA method
    atomicOr ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_bor<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // native CUDA method
    atomicOr ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

//------------------------------------------------------------------------------
// GB_atomic_band
//------------------------------------------------------------------------------

// bitwise: on uint [16,32,64]

template<> __device__ __inline__ void GB_atomic_band<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed & v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_band<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // native CUDA method
    atomicAnd ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_band<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // native CUDA method
    atomicAnd ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

//------------------------------------------------------------------------------
// GB_atomic_bxor
//------------------------------------------------------------------------------

// bitwise: on uint [16,32,64]

template<> __device__ __inline__ void GB_atomic_bxor<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, assumed ^ v) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_bxor<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // native CUDA method
    atomicXor ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_bxor<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // native CUDA method
    atomicXor ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

//------------------------------------------------------------------------------
// GB_atomic_bxnor
//------------------------------------------------------------------------------

// bitwise: on uint [16,32,64]

template<> __device__ __inline__ void GB_atomic_bxnor<uint16_t>(uint16_t* ptr, uint16_t val)
{
    unsigned short int *p = (unsigned short int *) ptr ;
    unsigned short int v = (unsigned short int) val ;
    unsigned short int assumed ;
    unsigned short int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, ~(assumed ^ v)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_bxnor<uint32_t>(uint32_t* ptr, uint32_t val)
{
    unsigned int *p = (unsigned int *) ptr ;
    unsigned int v = (unsigned int) val ;
    unsigned int assumed ;
    unsigned int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, ~(assumed ^ v)) ;
    }
    while (assumed != old) ;
}

template<> __device__ __inline__ void GB_atomic_bxnor<uint64_t>(uint64_t* ptr, uint64_t val)
{
    unsigned long long int *p = (unsigned long long int *) ptr ;
    unsigned long long int v = (unsigned long long int) val ;
    unsigned long long int assumed ;
    unsigned long long int old = *p ;
    do
    {
        // assume the old value
        assumed = old ;
        // modify it atomically:
        old = atomicCAS (p, assumed, ~(assumed ^ v)) ;
    }
    while (assumed != old) ;
}

//------------------------------------------------------------------------------
// GB_cuda_lock/unlock: set/clear a mutex for a critical section
//------------------------------------------------------------------------------

__device__ __inline__ void GB_cuda_lock (uint32_t *mutex)
{
    int old ;
    do
    {
        old = atomicCAS (mutex, 0, 1) ;
    }
    while (old == 1) ;
}

__device__ __inline__ void GB_cuda_unlock (uint32_t *mutex)
{
    int old ;
    do
    {
        old = atomicCAS (mutex, 1, 0) ;
    }
    while (old == 0) ;
}

