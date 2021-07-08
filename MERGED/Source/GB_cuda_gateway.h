//------------------------------------------------------------------------------
// GB_cuda_gateway.h: definitions for interface to GB_cuda_* functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CUDA gateway functions (DRAFT: in progress)

// This file can be #include'd into any GraphBLAS/Source file that needs to
// call a CUDA gateway function, or use the typedef defined below.  It is also
// #include'd in GraphBLAS/CUDA/GB_cuda.h, for use by the CUDA/GB_cuda_*.cu
// gateway functions.

// If GBCUDA is defined in GraphBLAS/CMakeLists.txt, then GraphBLAS can call
// the C-callable gateway functions defined in GraphBLAS/CUDA/*.cu source
// files.  If GBCUDA is not defined, then these functions are not called.  The
// typedef always appears, since it is part of the GB_Global struct, whether
// or not CUDA is used.

#ifndef GB_CUDA_GATEWAY_H
#define GB_CUDA_GATEWAY_H

#define GB_CUDA_MAX_GPUS 32

// The GPU is only used if the work is larger than the GxB_GPU_CHUNK.
// The default value of this parameter is GB_GPU_CHUNK_DEFAULT:
#define GB_GPU_CHUNK_DEFAULT (1024*1024)

#if defined ( GB_NVCC )
extern "C" {
#endif

//------------------------------------------------------------------------------
// GB_cuda_device: properties of each GPU in the system
//------------------------------------------------------------------------------

typedef struct
{
    char    name [256] ;
    size_t  total_global_memory ;
    int  number_of_sms ;
    int  compute_capability_major;
    int  compute_capability_minor;
    bool use_memory_pool;
    int  pool_size;             // TODO: should this be size_t?
    int  max_pool_size;         // TODO: should this be size_t?
    void *memory_resource;
}
GB_cuda_device ;

//------------------------------------------------------------------------------
// GB_ngpus_to_use: determine # of GPUs to use for the next computation
//------------------------------------------------------------------------------

static inline int GB_ngpus_to_use
(
    double work                 // total work to do
)
{

    // get the current GxB_GPU_CONTROL setting
    GrB_Desc_Value gpu_control = GB_Global_gpu_control_get ( ) ;

    // HACK:
    gpu_control = GxB_GPU_ALWAYS ;

    int gpu_count = GB_Global_gpu_count_get ( ) ;
    if (gpu_control == GxB_GPU_NEVER || gpu_count == 0)
    {
        // never use the GPU(s)
        return (0) ;
    }
    else if (gpu_control == GxB_GPU_ALWAYS)
    {
        // always use all available GPU(s)
        printf ("(using the GPU) ") ;
        return (gpu_count) ;
    }
    else
    {
        // use no more than max_gpus_to_use
        double gpu_chunk = GB_Global_gpu_chunk_get ( ) ;
        double max_gpus_to_use = floor (work / gpu_chunk) ;
        // but use no more than the # of GPUs available
        if (max_gpus_to_use > gpu_count) return (gpu_count) ;
        return ((int) max_gpus_to_use) ;
    }
}


//------------------------------------------------------------------------------
// GB_cuda_* gateway functions
//------------------------------------------------------------------------------

bool GB_cuda_get_device_count   // true if OK, false if failure
(
    int *gpu_count              // return # of GPUs in the system
) ;

bool GB_cuda_warmup (int device) ;

bool GB_cuda_get_device( int *device) ;

bool GB_cuda_set_device( int device) ;

bool GB_cuda_get_device_properties
(
    int device,
    GB_cuda_device *prop
) ;

void GB_stringify_semiring     // build a semiring (name and code)
(
    // output: (all of size at least GB_CUDA_STRLEN+1)
    char *semiring_macros,  // List of types and macro defs
    // input:
    GrB_Semiring semiring,  // the semiring to stringify
    bool flipxy,            // multiplier is: mult(a,b) or mult(b,a)
    GrB_Type ctype,         // the type of C
    GrB_Type mtype,         // the type of M, or NULL if no mask
    GrB_Type atype,         // the type of A
    GrB_Type btype,         // the type of B
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    int C_sparsity,         // sparsity structure of C
    int M_sparsity,         // sparsity structure of M
    int A_sparsity,         // sparsity structure of A
    int B_sparsity          // sparsity structure of B
);

void GB_enumify_semiring   // enumerate a semiring
(
    // output:
    uint64_t *scode,        // unique encoding of the entire semiring
    // input:
    GrB_Semiring semiring,  // the semiring to enumify
    bool flipxy,            // multiplier is: mult(a,b) or mult(b,a)
    GrB_Type ctype,         // the type of C
    GrB_Type mtype,         // the type of M, or NULL if no mask
    GrB_Type atype,         // the type of A
    GrB_Type btype,         // the type of B
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    int C_sparsity,         // sparsity structure of C
    int M_sparsity,         // sparsity structure of M
    int A_sparsity,         // sparsity structure of A
    int B_sparsity          // sparsity structure of B
);


bool GB_reduce_to_scalar_cuda_branch 
(
    const GrB_Monoid reduce,        // monoid to do the reduction
    const GrB_Matrix A,             // input matrix
    GB_Context Context
) ;

GrB_Info GB_reduce_to_scalar_cuda
(
    GB_void *s,
    const GrB_Monoid reduce,
    const GrB_Matrix A,
    GB_Context Context
) ;

GrB_Info GB_AxB_dot3_cuda           // C<M> = A'*B using dot product method
(
    GrB_Matrix C,                   // output matrix, static header
    const GrB_Matrix M,             // mask matrix
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Context Context
) ;

#if defined ( GB_NVCC )
}
#endif

#endif

