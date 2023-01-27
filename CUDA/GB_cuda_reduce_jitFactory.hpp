//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_reduce_jitFactory.hpp: kernel for reduction to scalar
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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
 */

//------------------------------------------------------------------------------

// Constructs an instance of the template/GB_jit_reduce.cuh kernel to reduce
// a GrB_Matrix to a scalar.

#ifndef GB_REDUCE_JITFACTORY_H
#define GB_REDUCE_JITFACTORY_H

#pragma once
#include "GB_cuda_reduce_factory.hpp"

/**
 * This file is responsible for picking all the parameters and what kernel
 * variaiton we will use for a given instance
 * - data types
 * - semiring types
 * - binary ops
 * - monoids
 *
 * Kernel factory says "Here's the actual instance I want you to build with the
 * given parameters"
 */

// Kernel jitifiers
class reduceFactory ;

// FIXME: what is spdotFactory?
template<typename T1, typename T2, typename T3> class spdotFactory ;

//------------------------------------------------------------------------------
// reduceFactory
//------------------------------------------------------------------------------

class reduceFactory
{

    //--------------------------------------------------------------------------
    // class properties
    //--------------------------------------------------------------------------

    std::string base_name = "GB_jit";
    std::string kernel_name = "reduce";

    int threads_per_block = 320 ;
    int work_per_thread = 256;
//  int number_of_sms = GB_Global_gpu_sm_get (0) ;

    GB_cuda_reduce_factory &reduce_factory_ ;

    public:

    //--------------------------------------------------------------------------
    // class constructor
    //--------------------------------------------------------------------------

    reduceFactory (GB_cuda_reduce_factory &myreducefactory) :
        reduce_factory_(myreducefactory) {}

    //--------------------------------------------------------------------------
    // GB_get_threads_per_block: determine # of threads in a threadBlock
    //--------------------------------------------------------------------------

    int GB_get_threads_per_block ( )
    {
        return threads_per_block ;
    }

    //--------------------------------------------------------------------------
    // GB_get_number_of_blocks: determine # of threadBlocks to use
    //--------------------------------------------------------------------------

    int GB_get_number_of_blocks
    (
        int64_t anvals     // # of entries in input matrix
    )
    {
        // FIXME: this is a lot of blocks.  Use a smaller number (cap at, say,
        // 64K), to simplify the non-atomic reductions
        return (anvals + work_per_thread*threads_per_block - 1) /
               (work_per_thread*threads_per_block) ;
    }

    //--------------------------------------------------------------------------
    // jitGridBlockLaunch:  construct and launch the GB_jit_reduce kernel
    //--------------------------------------------------------------------------

    // Note: this does assume the erased types are compatible w/ the monoid's
    // ztype (FIXME: what does this comment mean?)

    bool jitGridBlockLaunch     // FIXME: return GrB_Info
    (
        GrB_Matrix A,           // matrix to reduce to a scalar
        GB_void *output,        // output scalar (static on CPU), of size zsize
        GrB_Monoid monoid,      // monoid to use for the reducution
        cudaStream_t stream = 0 // stream to use, default stream 0
    )
    {
        GBURBLE ("\n(launch reduce factory) \n") ;

        // allocate and initialize zscalar
        size_t zsize = monoid->op->ztype->size ;
        // FIXME write a helper function to compute zscalar_size
        size_t zscalar_size = GB_IMAX (zsize, sizeof (uint16_t)) ;
        GB_void *zscalar = (GB_void *) rmm_wrap_malloc (zscalar_size) ;
        if (zscalar == NULL)
        {
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_cuda_upscale_identity (zscalar, monoid) ;

        jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
        filecache.getFile (reduce_factory_) ;

        auto rcode = std::to_string(reduce_factory_.rcode);

        std::string hashable_name = base_name + "_" + kernel_name;
        std::stringstream string_to_be_jitted ;
        string_to_be_jitted <<
        hashable_name << std::endl <<
        R"(#include ")" << jit::get_user_home_cache_dir() << "/"
        << reduce_factory_.filename << R"(")" << std::endl <<
        R"(#include ")" << hashable_name << R"(.cuh")" << std::endl;

        int64_t anvals = GB_nnz_held (A) ;

        // TODO: Use RMM!
        uint32_t *mutex ;
//      CU_OK(cudaMalloc((void**)&mutex, sizeof(int32_t)));
//      CU_OK(cudaMemsetAsync(mutex, 0, sizeof(int32_t), stream));
        // FIXED (but not tested): using RMM
        mutex = (uint32_t *) rmm_wrap_calloc (1, sizeof (uint32_t)) ;
        if (mutex == NULL)
        {
            // out of memory
            rmm_wrap_free (zscalar) ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        // determine kernel launch geometry
        int blocksz = GB_get_threads_per_block ( ) ;
        int gridsz = GB_get_number_of_blocks (anvals) ;
        dim3 grid (gridsz) ;
        dim3 block (blocksz) ;

        GBURBLE ("(cuda reduce launch %d threads in %d blocks)",
            blocksz, gridsz ) ;

        // construct and launch the kernel
        // FIXME: use GB_namify_problem to get the full name
        // FIXME: where does it go if it fails?  try/catch?
        jit::launcher(hashable_name + "_" + rcode,
                string_to_be_jitted.str(),
                header_names,
                compiler_flags,
                file_callback)  // FIXME: where is file_callback defined?
           .set_kernel_inst(  hashable_name ,
                { A->type->name, monoid->op->ztype->name })
           .configure(grid, block, SMEM, stream)
           .launch (A, zscalar, anvals, mutex) ;

        // synchronize before copying result to host
        CHECK_CUDA (cudaStreamSynchronize (stream)) ;

        // FIXME: sometimes we use CHECK_CUDA, sometimes CU_OK.  Need to
        // be consistent.  Also, if this method fails, mutex and zscalar
        // must be freed: we can do this in the CU_OK or CHECK_CUDA macros.
        // Or in a try/catch?

        // output = zscalar (but only the first zsize bytes of it)
        memcpy (output, zscalar, zsize) ;

        // free workspace and return result
//      CU_OK(cudaFree(mutex));
//      CHECK_CUDA (cudaFree (zscalar)) ;
        rmm_wrap_free (mutex) ;
        rmm_wrap_free (zscalar) ;

        return (GrB_SUCCESS) ;
    }
} ;

//------------------------------------------------------------------------------
// GB_cuda_reduce
//------------------------------------------------------------------------------

inline bool GB_cuda_reduce      // FIXME: return GrB_Info, not bool
(
    GB_cuda_reduce_factory &myreducefactory,    // reduction JIT factory
    GrB_Matrix A,               // matrix to reduce
    GB_void *output,            // result of size monoid->op->ztype->size
    GrB_Monoid monoid,          // monoid for the reduction
    cudaStream_t stream = 0     // stream to use
)
{
    reduceFactory rf(myreducefactory);
    GBURBLE ("(starting cuda reduce)" ) ;
    bool result = rf.jitGridBlockLaunch (A, output, monoid, stream) ;
    GBURBLE ("(ending cuda reduce)" ) ;
    return (result) ;
}

#endif

