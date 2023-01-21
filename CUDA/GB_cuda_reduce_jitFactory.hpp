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

/*
  Extended example for building on-the-fly kernels with C interface.
  Simple examples demonstrating different ways to load source code
    and call kernels.
 */

#ifndef GB_REDUCE_JITFACTORY_H
#define GB_REDUCE_JITFACTORY_H

#pragma once
#include "GB_cuda_reduce_factory.hpp"

/**
 * This file is responsible for picking all the parameters and what kernel variaiton we will use for a given instance
 * - data types
 * - semiring types
 * - binary ops
 * - monoids
 *
 * Kernel factory says "Here's the actual instance I want you to build with the given parameters"
 */

//bool GB_cuda_reduce(int64_t *index, void *in_data, void *output, unsigned int N, GrB_Monoid op);

//Kernel jitifiers
class reduceFactory ;
template<typename T1, typename T2, typename T3> class spdotFactory ;

//------------------------------------------------------------------------------
// reduceFactory
//------------------------------------------------------------------------------

class reduceFactory
{
    std::string base_name = "GB_jit";
    std::string kernel_name = "reduce";

    int threads_per_block = 320 ;
    int work_per_thread = 256;
    int number_of_sms = GB_Global_gpu_sm_get (0);

    GB_cuda_reduce_factory &reduce_factory_;

    public:

    reduceFactory (GB_cuda_reduce_factory &myreducefactory) :
        reduce_factory_(myreducefactory) {}

    int GB_get_threads_per_block()
    {
        return threads_per_block;
    }

    int GB_get_number_of_blocks(unsigned int N)
    {
        // FIXME: this is a lot of blocks.  Use a smaller number (cap at, say,
        // 64K), to simplify the non-atomic reductions
        return (N + work_per_thread*threads_per_block - 1) /
               (work_per_thread*threads_per_block) ;
    }

    // Note: this does assume the erased types are compatible w/ the monoid's
    // ztype (FIXME: what is this?)

    bool jitGridBlockLaunch
    (
        GrB_Matrix A,               // matrix to reduce to a scalar
        void *output,               // output scalar (static on the CPU)
        GrB_Monoid monoid,          // monoid to use for the reducution
        cudaStream_t stream = 0     // stream to use
    )
    {
        GBURBLE ("\n(launch reduce factory) \n") ;

        // allocate and initialize zscalar_result
        size_t zsize = monoid->op->ztype->size ;
        size_t zscalar_size = GB_IMAX (zsize, sizeof (int32_t)) ;
        GB_void identity_upscaled [zscalar_size] ;
        GB_cuda_upscale_identity (identity_upscaled, monoid) ;

        // FIXME: use unified shared memory functions, and the stream
        void *zscalar_result = NULL ;
        CHECK_CUDA (cudaMalloc (&zscalar_result, zscalar_size)) ;
        CHECK_CUDA (cudaMemcpy (zscalar_result, identity_upscaled,
            zscalar_size, cudaMemcpyHostToDevice)) ;

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

        //    bool is_sparse = GB_IS_SPARSE(A);
        //    int64_t N = is_sparse ? GB_nnz(A) : GB_NCOLS(A) * GB_NROWS(A);

        int64_t anz = GB_nnz_held (A) ;

        // TODO: Use RMM!
        int32_t *lock;
        CU_OK(cudaMalloc((void**)&lock, sizeof(int32_t)));
        CU_OK(cudaMemsetAsync(lock, 0, sizeof(int32_t), stream));

        int blocksz = GB_get_threads_per_block();
        int gridsz = GB_get_number_of_blocks(anz);
        dim3 grid(gridsz);
        dim3 block(blocksz);

        GBURBLE ("(cuda reduce launch %d threads in %d blocks)",
            blocksz, gridsz ) ;

        jit::launcher(hashable_name + "_" + rcode,
                string_to_be_jitted.str(),
                header_names,
                compiler_flags,
                file_callback)
           .set_kernel_inst(  hashable_name ,
                { A->type->name, monoid->op->ztype->name })
           .configure(grid, block, SMEM, stream)
           .launch( A, zscalar_result, anz, lock);

        // Need to synchronize before copying result to host
        CHECK_CUDA( cudaStreamSynchronize(stream) );

        // FIXME: use unified shared memory here instead
        // memcpy (output, zscalar_result, zsize) ;
        CHECK_CUDA (cudaMemcpy (output, zscalar_result,
            zsize, cudaMemcpyDeviceToHost)) ;

        CU_OK(cudaFree(lock));
        CHECK_CUDA (cudaFree (zscalar_result)) ;

        return (GrB_SUCCESS) ;
    }
} ;

//------------------------------------------------------------------------------

inline bool GB_cuda_reduce
(
    GB_cuda_reduce_factory &myreducefactory,
    GrB_Matrix A,
    void *output,               // statically allocated on the CPU
    GrB_Monoid monoid,
    cudaStream_t stream = 0
)
{
    reduceFactory rf(myreducefactory);
    GBURBLE ("(starting cuda reduce)" ) ;
    bool result = rf.jitGridBlockLaunch (A, output, monoid, stream) ;
    GBURBLE ("(ending cuda reduce)" ) ;
    return (result) ;
}

#endif

