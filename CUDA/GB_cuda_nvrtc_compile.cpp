//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_nvrtc_compile: compile a CUDA JIT kernel
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.h"

// compile the input file, of the form:
// %s/c/%02x/%s or [cache_path]/c/[bucket]/[kernel_name].cu

// and link the libary file as
// %s/lib/%02x/lib%s.so or [cache_path]/lib/[bucket]/lib[kernel_name].so

// then remove all temporary files.

// This function does not load the lib*.so file.

void GB_cuda_nvrtc_compile
(
    char *kernel_name,          // name of the kernel
    int bucket,                 // bucket to place the kernel in
    char *GB_jit_cache_path     // location of GraphBLAS cache
)
{

    // FIXME:  see GB_jitifyer_direct_compile for an example of how this
    // works on the CPU, for a CPU JIT kernel.

    GBURBLE ("(jit compiling cuda with nvrtc: %s/c/%02x/%s.cu) ",
        GB_jit_cache_path, bucket, kernel_name) ;

    // "#include  %s/GB_jit__cuda_reduce__ac1f881.cu", GB_jit_cache_path

}

