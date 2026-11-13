//------------------------------------------------------------------------------
// GB_cuda_memcpy_jit: JIT kernel for memcpy
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This JIT kernel only depends on the major/minor property of the target
// device, so in a typical system with a single type of GPU, only one of these
// JIT kernels will be compiled.

#include "memory/GB_cuda_memory.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_MEMCPY_PROTO ((*GB_jit_dl_function)) ;
}

GrB_Info GB_cuda_memcpy_jit
(
    // output:
    GB_void *dest,
    // input:
    const GB_void *src,
    const size_t n,             // # of bytes to transfer
    const GB_Operator op,       // GrB_IDENTITY_UINT8
    // CUDA stream and launch parameters:
    int device,
    cudaStream_t stream,
    int32_t gridsz
)
{

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_MEMCPY, /* C sparsity: */ GxB_FULL,
        /* C_is_matrix: */ false, /* C type: */ GrB_UINT8,
        /* pji_is_32: ignored; no matrix C: */ false, false, false,
        /* operator: */ op, /* flipij: */ false,
        /* A sparsity: */ GxB_FULL, /* A_is_matrix: */ false,
        /* A type: */ GrB_UINT8,
        /* pji_is_32: ignored; no matrix A: */ false, false, false,
        /* A iso: */ false, /* A zombies: */ false) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "cuda_memcpy",
        hash, &encoding, suffix, NULL, NULL,
        op, GrB_UINT8, GrB_UINT8, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (dest, src, n, device, stream, gridsz)) ;
}

