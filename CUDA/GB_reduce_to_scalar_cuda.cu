
//------------------------------------------------------------------------------
// GB_reduce_to_scalar_cuda.cu: reduce on the GPU with semiring 
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0
// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_cuda.h"

#include "templates/reduceWarp.cu.jit"
#include "test/semiringFactory.hpp"
#include "jitify.hpp"

GrB_Info GB_reduce_to_scalar_cuda
(
    GB_void *s,
    const GrB_Monoid reduce,
    const GrB_Matrix A,
    GB_Context Contetxt
)
{ 

    printf ("Hi I am %s :-)\n", __FILE__) ;

    // result = sum (Anz [0..anz-1]) using the GPU,
    // with a kernel that has ntasks = grid.x and blocksize = blockDim.x
    // nthreads = # of GPUs to use, but 1 for now
    // We have a workspace W of size ntasks.

    thread_local static jitify::JitCache kernel_cache;

    // stringified kernel specified above
    jitify::Program program= kernel_cache.program( templates_reduceWarp_cu, 0, 0,
        file_callback_plus);
    //{"--use_fast_math", "-I/usr/local/cuda/include"});

    int nnz = GB_NNZ( A ) ;
    int blocksize = 1024 ;
    int ntasks = ( nnz + blocksize -1) / blocksize ;

    dim3 grid(ntasks);
    dim3 block(blocksize);

    using jitify::reflection::type_of;
    program.kernel("reduceWarp")
                    .instantiate(type_of(*Ax))
                    .configure(grid, block)
                    .launch(Ax, W, anz);

    cudaDeviceSynchronize ( ) ;

    int64_t s = 0 ;
    for (int i = 0 ; i < ntasks ; i++)
    {
        s += W [i] ; 
    }

    (*result) = s ;

    return (GrB_SUCCESS) ;
}

