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

template <typename T>
__device__ void GB_atomic_write(T* ptr, T val);

template <typename T>
__device__ void GB_atomic_add(T* ptr, T val);

template <typename T>
__device__ void GB_atomic_min(T* ptr, T val);

template <typename T>
__device__ void GB_atomic_max(T* ptr, T val);

__device__ __inline__ void GB_cuda_lock (int32_t *lock) ;

__device__ __inline__ void GB_cuda_unlock (int32_t *lock) ;

//------------------------------------------------------------------------------
// GB_atomic_write
//------------------------------------------------------------------------------

// atomic write (8 bits, 16, 32, 64)

#if 0
// write these:
template<> __device__ __inline__ void GB_atomic_write<int8_t>(int8_t* ptr, int val)
template<> __device__ __inline__ void GB_atomic_write<int16_t>(int16_t* ptr, int val)
#endif

template<> __device__ __inline__ void GB_atomic_write<int32_t>(int32_t* ptr, int32_t val)
{
    // int32_t is the same as int
    atomicExch((int *) ptr, (int) val);
}

template<> __device__ __inline__ void GB_atomic_write<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // uint32_t is the same as unsigned int
    atomicExch((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_write<int64_t>(int64_t* ptr, int64_t val)
{
    // note the val is punned, it cannot be typecasted
    void *p = &val ;
    atomicExch ((unsigned long long int *) ptr, *((unsigned long long int *) p)) ;
}

template<> __device__ __inline__ void GB_atomic_write<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // uint64_t is the same as unsigned long long int
    atomicExch ((unsigned long long int *) ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_write<float>(float* ptr, float val)
{
    atomicExch(ptr, val) ;
}

template<> __device__ __inline__ void GB_atomic_write<double>(double* ptr, double val)
{
    // note the val is punned, it cannot be typecasted
    void *p = &val ;
    atomicExch ((unsigned long long int *) ptr, *((unsigned long long int *) p)) ;
}

//------------------------------------------------------------------------------
// GB_atomic_add for built-in types
//------------------------------------------------------------------------------

// types: int and uint [8,16,32,64], float, double, complex, complex double

#if 0
// write these:
template<> __device__ __inline__ void GB_atomic_add<int8_t>(int8_t* ptr, int val)
template<> __device__ __inline__ void GB_atomic_add<uint8_t>(uint8_t* ptr, int val)
template<> __device__ __inline__ void GB_atomic_add<int16_t>(int16_t* ptr, int val)
template<> __device__ __inline__ void GB_atomic_add<uint16_t>(uint16_t* ptr, int val)
#endif

template<> __device__ __inline__ void GB_atomic_add<int32_t>(int32_t* ptr, int32_t val)
{
    // int32_t is the same as int
    atomicAdd((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_atomic_add<uint32_t>(uint32_t* ptr, uint32_t val)
{
    // uint32_t is the same as unsigned int
    atomicAdd((unsigned int*)ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_add<int64_t>(int64_t* ptr, int64_t val)
{
    // note that val is punned, it cannot be typecasted
    void *p = &val ;
    atomicAdd((unsigned long long*)ptr, *((unsigned long long *) p)) ;
}

template<> __device__ __inline__ void GB_atomic_add<uint64_t>(uint64_t* ptr, uint64_t val)
{
    // uint64_t is the same as unsigned long long int
    atomicAdd((unsigned long long int*)ptr, (unsigned long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_add<float>(float* ptr, float val)
{
    // native CUDA method
    atomicAdd(ptr, val);
}

template<> __device__ __inline__ void GB_atomic_add<double>(double* ptr, double val)
{
    // native CUDA method
    atomicAdd(ptr, val);
}

#if 0
// write these:
template<> __device__ __inline__ void GB_atomic_add<float complex>(float complex* ptr, float  complexval)
{
    atomicAdd (ptr.real, val.real) ;
    atomicAdd (ptr.imag, val.imag) ;
}

template<> __device__ __inline__ void GB_atomic_add<double complex>(double complex* ptr, double complex val)
{
    atomicAdd (ptr.real, val.real) ;
    atomicAdd (ptr.imag, val.imag) ;
}
#endif

//------------------------------------------------------------------------------
// GB_atomic_times for built-in types ?
//------------------------------------------------------------------------------

// types: int and uint [8,16,32,64], float, double.
// no complex types.

// Is this possible?  There are no cuda atomic times functions.
// Maybe all need to be done with a lock.

//------------------------------------------------------------------------------
// GB_atomic_min
//------------------------------------------------------------------------------

// types: int and uint [8,16,32,64]
// no complex types, no float or double (use locks for those)

#if 0
// write these?  or use locks if they are too difficult
template<> __device__ __inline__ void GB_atomic_min<int8_t>(int8_t* ptr, int8_t val)
template<> __device__ __inline__ void GB_atomic_min<uint8_t>(uint8_t* ptr, uint8_t val)
template<> __device__ __inline__ void GB_atomic_min<int16_t>(int16_t* ptr, int16_t val)
template<> __device__ __inline__ void GB_atomic_min<uint16_t>(uint16_t* ptr, uint16_t val)
#endif

template<> __device__ __inline__ void GB_atomic_min<int32_t>(int32_t* ptr, int32_t val)
{
    atomicMin ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_atomic_min<uint32_t>(uint32_t* ptr, uint32_t val)
{
    atomicMin ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_min<int64_t>(int64_t* ptr, int64_t val)
{
    atomicMin ((long long int*)ptr, (long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_min<uint64_t>(uint64_t* ptr, uint64_t val)
{
    atomicMin ((unsigned long long int*)ptr, (unsigned long long int) val) ;
}

//------------------------------------------------------------------------------
// GB_atomic_max
//------------------------------------------------------------------------------

// types: int and uint [8,16,32,64]
// no complex types, no float or double (use locks for those)

#if 0
// write these?  or use locks if they are too difficult
template<> __device__ __inline__ void GB_atomic_max<int8_t>(int8_t* ptr, int8_t val)
template<> __device__ __inline__ void GB_atomic_max<uint8_t>(uint8_t* ptr, uint8_t val)
template<> __device__ __inline__ void GB_atomic_max<int16_t>(int16_t* ptr, int16_t val)
template<> __device__ __inline__ void GB_atomic_max<uint16_t>(uint16_t* ptr, uint16_t val)
#endif

template<> __device__ __inline__ void GB_atomic_max<int32_t>(int32_t* ptr, int32_t val)
{
    atomicMax ((int *) ptr, (int) val) ;
}

template<> __device__ __inline__ void GB_atomic_max<uint32_t>(uint32_t* ptr, uint32_t val)
{
    atomicMax ((unsigned int *) ptr, (unsigned int) val) ;
}

template<> __device__ __inline__ void GB_atomic_max<int64_t>(int64_t* ptr, int64_t val)
{
    atomicMax ((long long int*)ptr, (long long int) val) ;
}

template<> __device__ __inline__ void GB_atomic_max<uint64_t>(uint64_t* ptr, uint64_t val)
{
    atomicMax ((unsigned long long int*)ptr, (unsigned long long int) val) ;
}

//------------------------------------------------------------------------------
// GB_atomic_lor, land, lxor, lxnor
//------------------------------------------------------------------------------

// bool:  or, and, xor, xnor (8-bit bool)

// how??  CUDA supports only 32-bit and 64-bit bitwise or, and, xor

//------------------------------------------------------------------------------
// GB_atomic_bor, band, bxor, bxnor
//------------------------------------------------------------------------------

// bitwise:     on uint [8,16,32,64]
//      bor
//      band
//      bxor
//      bxnor       use lock for this?


//------------------------------------------------------------------------------
// GB_cuda_lock/unlock: set/clear a lock for a critical section
//------------------------------------------------------------------------------

__device__ __inline__ void GB_cuda_lock (int32_t *lock)
{
    while (atomicExch ((int *) lock, (int) 1) != 0) ;
}

__device__ __inline__ void GB_cuda_unlock (int32_t *lock) 
{
    lock = 0 ;
    // or this ?  It might be safer:
    // GB_atomic_write <int32_t> (lock, 0) ;
}

