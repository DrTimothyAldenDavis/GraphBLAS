//------------------------------------------------------------------------------
// GraphBLAS/CUDA/Template/GB_cuda_kernel.cuh: definitions for CUDA kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd into all device functions for CUDA JIT kernels for
// GraphBLAS.  It provides a subset of GraphBLAS.h and GB.h, plus other
// definitions.  It is not used on the host.

#pragma once

//------------------------------------------------------------------------------
// C++ and CUDA #include files
//------------------------------------------------------------------------------

#include <limits>
#include <type_traits>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <cub/block/block_scan.cuh>
#include <cooperative_groups.h>
using namespace cooperative_groups ;

//------------------------------------------------------------------------------
// CUDA kernel definitions
//------------------------------------------------------------------------------

#define GB_CUDA_KERNEL

#undef  ASSERT
#define ASSERT(x)

// FIXME: move this to where it's used
#define chunksize 128 

//------------------------------------------------------------------------------
// NVIDIA warp size
//------------------------------------------------------------------------------

#define WARPSIZE 32
#define LOG2_WARPSIZE 5

//------------------------------------------------------------------------------

#ifndef INFINITY
#define INFINITY (std::numeric_limits<double>::max())
#endif

// for internal static inline functions
#undef  GB_STATIC_INLINE
#define GB_STATIC_INLINE static __device__ __inline__

//------------------------------------------------------------------------------
// subset of GraphBLAS.h
//------------------------------------------------------------------------------

#include "GraphBLAS_h_subset.cuh"

//------------------------------------------------------------------------------
// subset of GB.h
//------------------------------------------------------------------------------

#include "GB_h_subset.cuh"

//------------------------------------------------------------------------------
// GB_search_for_vector_device
//------------------------------------------------------------------------------

static __device__ __inline__ int64_t GB_search_for_vector_device
(
    const int64_t p,                // search for vector k that contains p
    const int64_t *restrict Ap,     // vector pointers to search
    int64_t kleft,                  // left-most k to search
    int64_t anvec,                  // Ap is of size anvec+1
    int64_t avlen                   // A->vlen
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (Ap == NULL)
    { 
        // A is full or bitmap
        ASSERT (p >= 0 && p < avlen * anvec) ;
        return ((avlen == 0) ? 0 : (p / avlen)) ;
    }

    // A is sparse
    ASSERT (p >= 0 && p < Ap [anvec]) ;

    //--------------------------------------------------------------------------
    // search for k
    //--------------------------------------------------------------------------

    int64_t k = kleft ;
    int64_t kright = anvec ;
    bool found ;
    GB_SPLIT_BINARY_SEARCH (p, Ap, k, kright, found) ;
    if (found)
    {
        // Ap [k] == p has been found, but if k is an empty vector, then the
        // next vector will also contain the entry p.  In that case, k needs to
        // be incremented until finding the first non-empty vector for which
        // Ap [k] == p.
        ASSERT (Ap [k] == p) ;
        while (k < anvec-1 && Ap [k+1] == p)
        { 
            k++ ;
        }
    }
    else
    { 
        // p has not been found in Ap, so it appears in the middle of Ap [k-1]
        // ... Ap [k], as computed by the binary search.  This is the range of
        // entries for the vector k-1, so k must be decremented.
        k-- ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // The entry p must reside in a non-empty vector.
    ASSERT (k >= 0 && k < anvec) ;
    ASSERT (Ap [k] <= p && p < Ap [k+1]) ;

    return (k) ;
}

//------------------------------------------------------------------------------
// final #include files
//------------------------------------------------------------------------------

#include "GB_cuda_error.hpp"
#include "GB_printf_kernels.h"
#include "GB_cuda_atomics.cuh"
#include "GB_hash.h"
#include "GB_hyper_hash_lookup.h"

extern "C"
{
    #include "GB_werk.h"
    #include "GB_callback.h"
}

