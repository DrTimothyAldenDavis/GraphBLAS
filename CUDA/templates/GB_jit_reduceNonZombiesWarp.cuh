//------------------------------------------------------------------------------
// template/GB_jit_reduceNonZombiesWarp.cu
//------------------------------------------------------------------------------

// The GB_jit_reduceNonZombiesWarp CUDA kernel reduces a GrB_Matrix A 
// of any type T_A, to a scalar of type T_Z.  Each threadblock
// (blockIdx.x) reduces its portion of Ax to a single scalar, and then
// atomics are used across the threadblocks.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x.
// Each threadblock owns s*8 contiguous items in the input data.

// Thus, threadblock b owns Ax [b*s*8 ... min(n,(b+1)*s*8-1)].  Its job
// is to reduce this data to a scalar, and write it to its scalar.

// If the reduction is done on the GPU, A will never be iso-valued.

#define GB_CUDA_KERNEL
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

// FIXME: assumes 32 threads in a warp.  Is that safe?
// FIXME: why rcode?  It is not needed here (it is uint64_t not int)

template< typename T_Z, int tile_sz, int rcode>
__inline__ __device__ 
T_Z GB_warp_Reduce( thread_block_tile<tile_sz> g, T_Z val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[k] to sum[lane+k]
    T_Z fold = g.shfl_down( val, 16);
    GB_ADD( val, val, fold );
    fold = g.shfl_down( val, 8);
    GB_ADD( val, val, fold );
    fold = g.shfl_down( val, 4);
    GB_ADD( val, val, fold );
    fold = g.shfl_down( val, 2);
    GB_ADD( val, val, fold );
    fold = g.shfl_down( val, 1);
    GB_ADD( val, val, fold );
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// GB_block_Reduce: reduce across all warps into a single scalar
//------------------------------------------------------------------------------

// FIXME: warpSize parameter ignored?  Only works for warp of 32?

template<typename T_Z, int warpSize, int rcode>
__inline__ __device__
T_Z GB_block_Reduce(thread_block g, T_Z val)
{
    static __shared__ T_Z shared[warpSize]; // Shared mem for 32 partial sums
    int lane = threadIdx.x & 31 ; // % warpSize;
    int wid  = threadIdx.x >> 5 ; // / warpSize;
    thread_block_tile<warpSize> tile = tiled_partition<warpSize>( g );

    // FIXME: Figure out how to use graphblas-specific INFINITY macro
    #ifndef INFINITY
    #define INFINITY std::numeric_limits<double>::max()
    #endif

    // Each warp performs partial reduction
    val = GB_warp_Reduce<T_Z, warpSize, rcode>( tile, val);

    // Wait for all partial reductions
    if (lane==0)
    { 
        shared[wid] = val; // Write reduced value to shared memory
    }
    this_thread_block().sync();     // Wait for all partial reductions
    GB_DECLARE_MONOID_IDENTITY (identity) ;

    val = (threadIdx.x < (blockDim.x / warpSize) ) ? shared[lane] : identity ;

    // Final reduce within first warp
    val = GB_warp_Reduce<T_Z, warpSize, rcode>( tile, val);
    return val;
}

//------------------------------------------------------------------------------
// reduceNonZombiesWarp: reduce all entries in a matrix to a single scalar
//------------------------------------------------------------------------------

// FIXME: do not use a GrB_Scalar result, since it must be an array of size
// grid.x (# threadblocks, or gridDim.x?)

// FIXME: handle bitmap case

template< typename T_A, typename T_Z, int rcode, bool atomic_reduce = true>
__global__ void reduceNonZombiesWarp    // FIXME: rename
(
    GrB_Matrix A,
    GrB_Scalar R,   // array of size grid.x if atomic_reduce==false and
                    // size 1 if atomic_reduce==true ???
    int64_t N,      // number of entries for sparse, size of x array for
                    // full/bitmap.  FIXME: do not use N, looks like a #define
                    // FIXME: do not pass in N
    bool is_sparse  // FIXME: remove this
)
{

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    #if GB_A_IS_SPARSE || GB_A_IS_HYPERSPARSE
    int64_t anz = A->nvals ;              // may include zombies
    #else
    int64_t anz = A->vlen * A->vdim ;
    #endif

    // FIXME: move this to GB_cuda_kernel.h
    #ifndef INFINITY
    #define INFINITY std::numeric_limits<T_Z>::max()
    #endif

    const T_A *__restrict__ Ax = (T_A *) A->x ;

    // each thread reduces its result into sum, of type T_Z
    GB_DECLARE_MONOID_IDENTITY (sum) ;  // FIXME: rename this scalar

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
                T_Z aij ;
                GB_GETA (aij, Ax, p, false) ;   // aij = (T_Z) Ax [p]
                GB_ADD( sum, sum, aij );
            }
        }
        else
        {
            // no zombies present
            for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
                p < anz ;
                p += blockDim.x * gridDim.x)
            {
                T_Z aij ;
                GB_GETA (aij, Ax, p, false) ;   // aij = (T_Z) Ax [p]
                GB_ADD( sum, sum, aij );
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
            T_Z aij ;
            GB_GETA (aij, Ax, p, false) ;       // aij = (T_Z) Ax [p]
            GB_ADD( sum, sum, aij );
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
            T_Z aij ;
            GB_GETA (aij, Ax, p, false) ;   // aij = (T_Z) Ax [p]
            GB_ADD( sum, sum, aij );
        }
    }
    #endif

    this_thread_block().sync(); 

    //--------------------------------------------------------------------------
    // phase 2: each threadblock reduces all threads into a scalar
    //--------------------------------------------------------------------------

    // this assumes blockDim is a multiple of 32
    // FIXME: 32 is hard coded.  Is that safe?

    sum = GB_block_Reduce< T_A, 32, rcode >( this_thread_block(), sum) ;
    this_thread_block().sync(); 

    //--------------------------------------------------------------------------
    // phase 3: reduce across threadblocks, or punt to the CPU
    //--------------------------------------------------------------------------

    // scalar result
    T_Z *g_odata = (T_Z *) R->x ;

#if 0
    // FIXME: is this OK?
    #if !GB_CUDA_HAS_ATOMC
    int32_t __global__ lock = 0 ;
    #endif
#endif

    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
#if 0
        // NEW:
        #if GB_CUDA_HAS_ATOMC
            // lock free atomic operation
            GB_CUDA_ATOMIC <T_Z> (g_odata, sum) ;
        #else
            // the monoid does not have an atomic variant; use a lock
            GB_cuda_lock (&lock) ;
            GB_ADD (g_odata, g_odata, sum) ;
            GB_cuda_unlock (&lock) ;
        #endif
#endif

        // OLD:
        // all threadblocks reduce their result via an atomic
        GB_atomic_add<T_Z>(g_odata, sum);
    }
}

