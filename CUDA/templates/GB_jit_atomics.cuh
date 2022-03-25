#pragma once

template <typename T>
__device__ void atomic_add(T* ptr, T val);

template<> __device__ __inline__ void atomic_add<int>(int* ptr, int val) { atomicAdd(ptr, val); }
template<> __device__ __inline__ void atomic_add<int64_t>(int64_t* ptr, int64_t val) { atomicAdd((unsigned long long*)ptr, (unsigned long long)val); }

