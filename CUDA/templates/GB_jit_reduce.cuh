//------------------------------------------------------------------------------
// template/GB_jit_reduce.cu
//------------------------------------------------------------------------------

// The GB_jit_reduce CUDA kernel reduces a GrB_Matrix A of any type T_A, to a
// scalar of type T_Z.  Each threadblock (blockIdx.x) reduces its portion of Ax
// to a single scalar, and then atomics are used across the threadblocks.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x.
// Each threadblock owns s*8 contiguous items in the input data.

// Thus, threadblock b owns Ax [b*s*8 ... min(n,(b+1)*s*8-1)].  Its job
// is to reduce this data to a scalar, and write it to its scalar.

// If the reduction is done on the GPU, A will never be iso-valued.

#include <limits>
#include <type_traits>
#include "GB_cuda_kernel.h"
#include "GB_cuda_atomics.cuh"
#include <cstdint>
#include <cooperative_groups.h>

using namespace cooperative_groups;

//------------------------------------------------------------------------------
// GB_warp_Reduce: reduce all entries in a warp to a single scalar
//------------------------------------------------------------------------------

// GB_warp_Reduce assumes WARPSIZE is 32 threads.

template<typename T_Z>
__inline__ __device__
T_Z GB_warp_Reduce( thread_block_tile<WARPSIZE> g, T_Z val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[k] to sum[lane+k]

    // FIXME: doesn't work unless sizeof(T_Z) <= 32 bytes

    T_Z fold = g.shfl_down ( val, 16) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 8) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 4) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 2) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 1) ;
    GB_ADD ( val, val, fold ) ;
    return (val) ; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// GB_block_Reduce: reduce across all warps into a single scalar
//------------------------------------------------------------------------------

template<typename T_Z>
__inline__ __device__
T_Z GB_block_Reduce(thread_block g, T_Z val)
{
    static __shared__ T_Z shared [WARPSIZE] ;
    int lane = threadIdx.x & (WARPSIZE-1) ;
    int wid  = threadIdx.x >> LOG2_WARPSIZE ;
    thread_block_tile<WARPSIZE> tile = tiled_partition<WARPSIZE>( g ) ;

    // Each warp performs partial reduction
    val = GB_warp_Reduce<T_Z>( tile, val) ;

    // Wait for all partial reductions
    if (lane == 0)
    {
        shared [wid] = val ; // Write reduced value to shared memory
    }
    this_thread_block().sync() ;     // Wait for all partial reductions
    GB_DECLARE_MONOID_IDENTITY (identity) ;

    val = (threadIdx.x < (blockDim.x >> LOG2_WARPSIZE)) ? shared [lane] : identity ;

    // Final reduce within first warp
    val = GB_warp_Reduce<T_Z>( tile, val) ;
    return (val) ;
}

//------------------------------------------------------------------------------
// GB_jit_reduce: reduce all entries in a matrix to a single scalar
//------------------------------------------------------------------------------

template< typename T_A, typename T_Z>
__global__ void GB_jit_reduce
(
    GrB_Matrix A,   // matrix to reduce
    void *zscalar,  // scalar result, at least sizeof (uint16_t)
    int64_t anz,    // # of entries in A: anz = GB_nnz_held (A)
    uint32_t *mutex // for monoids that need it
)
{

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    const T_A *__restrict__ Ax = (T_A *) A->x ;

    // each thread reduces its result into sum, of type T_Z
    GB_DECLARE_MONOID_IDENTITY (sum) ;  // FIXME: rename this scalar

    // On input, zscalar is already initialized to the monoid identity value.
    // If T_Z has size less than 4 bytes, zscalar has been upscaled to 4 bytes.

    //--------------------------------------------------------------------------
    // phase 1: each thread reduces a part of the matrix to its own scalar
    //--------------------------------------------------------------------------

    #if GB_A_IS_SPARSE || GB_A_IS_HYPERSPARSE
    {

        //----------------------------------------------------------------------
        // A is sparse or hypersparse
        //----------------------------------------------------------------------

        // FUTURE: the check for zombies could be decided at compile-time

        if (A->nzombies > 0)
        {
            // check for zombies during the reduction
            const int64_t *__restrict__ Ai = A->i ;
            for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
                p < anz ;
                p += blockDim.x * gridDim.x)
            {
                if (Ai [p] < 0) continue ;      // skip zombies
                GB_DECLAREA (aij) ;
                GB_GETA (aij, Ax, p, false) ;   // aij = (T_Z) Ax [p]
                GB_ADD ( sum, sum, aij ) ;
            }
        }
        else
        {
            // no zombies present
            for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
                p < anz ;
                p += blockDim.x * gridDim.x)
            {
                GB_DECLAREA (aij) ;
                GB_GETA (aij, Ax, p, false) ;   // aij = (T_Z) Ax [p]
                GB_ADD ( sum, sum, aij ) ;
            }
        }

    }
    #elif GB_A_IS_FULL
    {

        //----------------------------------------------------------------------
        // A is full
        //----------------------------------------------------------------------

        for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
            p < anz ;
            p += blockDim.x * gridDim.x)
        {
            GB_DECLAREA (aij) ;
            GB_GETA (aij, Ax, p, false) ;       // aij = (T_Z) Ax [p]
            GB_ADD ( sum, sum, aij ) ;
        }

    }
    #else
    {

        //----------------------------------------------------------------------
        // A is bitmap
        //----------------------------------------------------------------------

        const uint8_t *__restrict__ Ab = A->b ;
        for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
            p < anz ;
            p += blockDim.x * gridDim.x)
        {
            if (Ab [p] == 0) continue ;     // skip if entry not in bitmap
            GB_DECLAREA (aij) ;
            GB_GETA (aij, Ax, p, false) ;   // aij = (T_Z) Ax [p]
            GB_ADD ( sum, sum, aij ) ;
        }
    }
    #endif

    this_thread_block().sync() ;

    //--------------------------------------------------------------------------
    // phase 2: each threadblock reduces all threads into a scalar
    //--------------------------------------------------------------------------

    sum = GB_block_Reduce< T_Z >( this_thread_block(), sum) ;
    this_thread_block().sync() ;

    //--------------------------------------------------------------------------
    // phase 3: reduce across threadblocks
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0)
    {
        #if GB_HAS_CUDA_ATOMIC

            // cast the result to the CUDA atomic type, and reduce
            // atomically to the global zscalar
            GB_CUDA_ATOMIC_TYPE *z = (GB_CUDA_ATOMIC_TYPE *) zscalar ;
            GB_CUDA_ATOMIC_TYPE zsum = (GB_CUDA_ATOMIC_TYPE) sum ;
            GB_CUDA_ATOMIC <GB_CUDA_ATOMIC_TYPE> (z, zsum) ;

        #else

            // FIXME:  use another kind of reduction.  Write the kth
            // threadblock result to Result [k], and use another kernel launch?
            // The Result array should be padded for 8-bit and maybe 16-bit
            // types, even for user-defined types, so that each threadblock
            // writes at least one single word.  Limit the # of threadblocks to
            // some upperbound, say 64K.

            T_Z *z = (T_Z *) zscalar ;
            GB_cuda_lock (mutex) ;
            GB_ADD (z, z, sum) ;
            GB_cuda_unlock (mutex) ;

        #endif
    }
}

