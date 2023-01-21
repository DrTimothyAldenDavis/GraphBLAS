//------------------------------------------------------------------------------
// GB_reduce_to_scalar_cuda.cu: reduce on the GPU with semiring 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

extern "C"
{
#include "GB_reduce.h"
}

#include "GB_cuda.h"
#include "GB_jit_cache.h"
#include "GB_cuda_common_jitFactory.hpp"
#include "GB_cuda_reduce_jitFactory.hpp"
#include "GB_cuda_type_wrap.hpp"

GrB_Info GB_reduce_to_scalar_cuda
(
    // output:
    GB_void *s,                 // note: statically allocated on CPU stack
    // input:
    const GrB_Monoid reduce,
    const GrB_Matrix A
)
{

    // FIXME: use the stream pool
    cudaStream_t stream ;
    CHECK_CUDA (cudaStreamCreate (&stream)) ;

    //----------------------------------------------------------------------
    // reduce C to a scalar
    //----------------------------------------------------------------------

    // FIXME: check error conditions (out of memory, etc)
    GB_cuda_reduce_factory myreducefactory ;
    myreducefactory.reduce_factory (reduce, A) ;

    GB_cuda_reduce (myreducefactory, A, s, reduce, stream) ;

    CHECK_CUDA (cudaStreamSynchronize (stream)) ;
    CHECK_CUDA (cudaStreamDestroy (stream)) ;

    return (GrB_SUCCESS) ;
}

