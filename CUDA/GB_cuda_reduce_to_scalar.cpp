//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_reduce_to_scalar: reduce on the GPU with semiring 
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix A to a scalar s, or to a smaller matrix V if the GPU was
// only able to do a partial reduction.  This case occurs if the GPU does not
// cannot do an atomic update for the monoid.  To handle this case, the GPU
// returns a full GrB_Matrix V, of size gridsize-by-1, with one entry per
// threadblock.  Then GB_reduce_to_scalar on the CPU sees this V as the result,
// and calls itself recursively to continue the reduction.

#define GB_FREE_ALL             \
{                               \
    rmm_wrap_free (zscalar) ;   \
    GB_Matrix_free (&V) ;       \
}

#include "GB_cuda_reduce.h"

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

    GB_void *zscalar = NULL ;
    GrB_Matrix V = NULL ;
    (*V_handle) = NULL ;
    GrB_Info info ;

    //--------------------------------------------------------------------------
    // create the stream
    //--------------------------------------------------------------------------

    // FIXME: use the stream pool
    cudaStream_t stream = 0 ;
    CHECK_CUDA (cudaStreamCreate (&stream)) ;

    //--------------------------------------------------------------------------
    // determine problem characteristics and allocate worksbace
    //--------------------------------------------------------------------------

    int threads_per_block = 320 ;
    int work_per_thread = 256;
//  int number_of_sms = GB_Global_gpu_sm_get (0) ;

    GBURBLE ("\n(launch reduce factory) \n") ;

    GrB_Type ztype = monoid->op->ztype ;
    size_t zsize = ztype->size ;

    // determine kernel launch geometry
    int64_t anvals = GB_nnz_held (A) ;
    int blocksz = threads_per_block ;
    int gridsz =
        // FIXME: this is a lot of blocks.  Use a smaller number (cap at,
        // say, 64K), to simplify the non-atomic reductions
        (anvals + work_per_thread*threads_per_block - 1) /
               (work_per_thread*threads_per_block) ;

    // FIXME: GB_enumify_reduce is called twice: here (to get has_cheeseburger)
    // and in GB_cuda_reduce_to_scalar_jit.  Can we just call it once?  One
    // solution: The code from here to the call to GB_cuda_reduce_to_scalar_jit
    // could be added to the GB_cuda_reduce_to_scalar_jit function itself.

    uint64_t rcode ;
    GB_enumify_reduce (&rcode, monoid, A) ;
    bool has_cheeseburger = GB_RSHIFT (rcode, 27, 1) ;
    GBURBLE ("has_cheeseburger %d\n", has_cheeseburger) ;

    // determine the kind of reduction: partial (to &V), or complete
    // (to the scalar output)
    if (has_cheeseburger)
    {
        // the kernel launch can reduce A to zscalar all by itself
        // allocate and initialize zscalar (upscaling it to at least 32 bits)
        size_t zscalar_size = GB_IMAX (zsize, sizeof (uint32_t)) ;
        (GB_void *) rmm_wrap_malloc (zscalar_size) ;
        zscalar = (GB_void *) rmm_wrap_malloc (zscalar_size) ;
        if (zscalar == NULL)
        {
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_cuda_upscale_identity (zscalar, monoid) ;
    }
    else
    {
        // allocate a full GrB_Matrix V for the partial result, of size
        // gridsz-by-1, and of type ztype.  V is allocated but not
        // initialized.
        GrB_Info info = GB_new_bix (&V, ztype, gridsz, 1, GB_Ap_null,
            true, GxB_FULL, false, 0, -1, gridsz, true, false) ;
        if (info != GrB_SUCCESS)
        {
            // out of memory
            return (info) ;
        }
    }

    GBURBLE ("(cuda reduce launch %d threads in %d blocks)",
        blocksz, gridsz ) ;

    //--------------------------------------------------------------------------
    // reduce C to a scalar via the CUDA JIT
    //--------------------------------------------------------------------------

    // FIXME: could let the function itself allocate zscalar and V:
    // GB_cuda_reduce_to_scalar_jit (&zscalar, &V, monoid, A,
    //     stream, gridsz, blocksz) ;
    GB_cuda_reduce_to_scalar_jit (zscalar, V, monoid, A,
        stream, gridsz, blocksz) ;

    //--------------------------------------------------------------------------
    // get result
    //--------------------------------------------------------------------------

    // FIXME: sometimes we use CHECK_CUDA, sometimes CU_OK.  Need to
    // be consistent.  Also, if this method fails, zscalar
    // must be freed: we can do this in the CU_OK or CHECK_CUDA macros.
    // Or in a try/catch?

    // FIXME: this could be folded into GB_cuda_reduce_to_scalar_jit:

    if (has_cheeseburger)
    {
        // return the scalar result
        // s = zscalar (but only the first zsize bytes of it)
        memcpy (s, zscalar, zsize) ;
        rmm_wrap_free (zscalar) ;
    }
    else
    {
        // return the partial reduction
        (*V_handle) = V ;
    }

    //--------------------------------------------------------------------------
    // synchronize before copying result to host
    //--------------------------------------------------------------------------

    CHECK_CUDA (cudaStreamSynchronize (stream)) ;
    CHECK_CUDA (cudaStreamDestroy (stream)) ;
    return (GrB_SUCCESS) ;
}

