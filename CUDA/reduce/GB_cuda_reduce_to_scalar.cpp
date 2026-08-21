//------------------------------------------------------------------------------
// CUDA/reduce/GB_cuda_reduce_to_scalar: reduce on the GPU with semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix A to a scalar s, or to a smaller vector V if the GPU was
// only able to do a partial reduction.  This case occurs if the GPU does not
// cannot do an atomic update for the monoid.  To handle this case, the GPU
// returns a full GrB_Matrix V, of size gridsize-by-1, with one entry per
// threadblock.  Then GB_reduce_to_scalar on the CPU sees this V as the result,
// and calls itself recursively to continue the reduction.

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                   \
{                                                           \
    GB_FREE_MEMORY (&zscalar, zscalar_mem) ;                \
}

#define GB_FREE_ALL                                         \
{                                                           \
    GB_FREE_WORKSPACE ;                                     \
    GB_Matrix_free (&V) ;                                   \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

#include "reduce/GB_cuda_reduce.hpp"

GrB_Info GB_cuda_reduce_to_scalar
(
    // output:
    GB_void *s,                 // note: statically allocated on CPU stack; if
                                // the result is in s then V is NULL.
    GrB_Matrix *V_handle,       // partial result if unable to reduce to scalar;
                                // NULL if result is in s.
    // input:
    const GrB_Monoid monoid,
    const GrB_Matrix A
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    int device = 0 ;    // fixme
    int data_arena = GrB_DEFAULT ;  // fixme: will depend on device id
    uint64_t mem = GB_mem (data_arena, 0) ;

    GB_void *zscalar = NULL ;
    uint64_t zscalar_mem = mem ;
    GrB_Matrix V = NULL ;
    (*V_handle) = NULL ;
    GrB_Info info = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // create the stream
    //--------------------------------------------------------------------------

    cudaStream_t stream = nullptr ;
    GB_OK (GB_cuda_stream_pool_acquire (&stream)) ;

    //--------------------------------------------------------------------------
    // determine problem characteristics and allocate worksbace
    //--------------------------------------------------------------------------

    int work_per_thread = 256 ;     // work each thread does in a single block
    int number_of_sms = GB_Global_gpu_sm_get (device) ;

    GrB_Type ztype = monoid->op->ztype ;
    size_t zsize = ztype->size ;

    // determine kernel launch geometry
    int64_t anvals = GB_nnz_held (A) ;
    int64_t work_per_block = work_per_thread * GB_CUDA_REDUCE_BLOCKDIM ;
    // gridsz = min (ceil (anvals / work_per_block), number_of_sms * 256)
    int64_t raw_gridsz = GB_ICEIL (anvals, work_per_block) ;
    raw_gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    int gridsz = (int) raw_gridsz ;

    uint64_t rcode ;
    GB_enumify_reduce (&rcode, monoid, A) ;
    bool has_cheeseburger = GB_RSHIFT (rcode, 16, 1) ;


    // determine the kind of reduction: partial (to &V), or complete
    // (to the scalar output)
    if (has_cheeseburger)
    {
        // has_cheeseburger is true if CUDA has an atomic operation that can be
        // used for the monoid.  If so, the kernel launch can reduce A to
        // zscalar all by itself allocate and initialize zscalar (upscaling it
        // to at least 32 bits)
        size_t zscalar_space = GB_IMAX (zsize, sizeof (uint32_t)) ;
        zscalar = (GB_void *) GB_MALLOC_MEMORY (1, zscalar_space,
            &zscalar_mem) ;
        if (zscalar == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_cuda_upscale_identity (zscalar, monoid) ;
    }
    else
    {
        // has_cheeseburger is false, which means there is no CUDA atomic
        // operator that matches the monoid.  If has_cheeseburger is false, the
        // threadblocks reduce their part of the matrix or vector to a single
        // scalar, placing their result in the vector V.  The vector V is full,
        // with the ztype, of length gridsz-by-1 (the number of threadblocks in
        // the kernel launch).  V is allocated but not initialized.
        GB_OK (GB_new_bix (&V, ztype, gridsz, 1, GB_ph_null,
            /* is_csc: */ true, /* sparsity: */ GxB_FULL,
            /* bitmap_calloc: */ false, /* hyper_switch: */ 0,
            /* plen: */ -1, /* nzmax: */ gridsz, /* numeric: */ true,
            /* iso: */ false, /* pji_is_32: */ false, false, false,
            data_arena, data_arena)) ;
    }

    GBURBLE ("(cuda reduce: %d threads per block; %d blocks; CUDA has"
        " atomic op: %d) ", GB_CUDA_REDUCE_BLOCKDIM, gridsz, has_cheeseburger) ;

    //--------------------------------------------------------------------------
    // reduce C to a scalar via the CUDA JIT
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_reduce_to_scalar_jit (zscalar, V, monoid, A,
        stream, gridsz)) ;

    //--------------------------------------------------------------------------
    // return result and release the stream
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_stream_pool_release (&stream)) ;

    if (has_cheeseburger)
    {
        // return the scalar result
        // s = zscalar (but only the first zsize bytes of it)
        memcpy (s, zscalar, zsize) ;
    }
    else
    {
        // return the partial reduction; the caller must continue the
        // reduction to reduce V to a single scalar.
        (*V_handle) = V ;

        // FUTURE: If the monoid is terminal, V will contain uninitialized
        // values.  It should be initialized first, by the JIT kernel
        // or above, after creating it.  Alternatively, if the reduction
        // triggers the early-termination flag, this could be returned to
        // the caller, and V would not be used.  Instead, the final scalar
        // result would be the terminal value of the monoid.
    }

    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

