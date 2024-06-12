using namespace cooperative_groups ;

#define tile_sz 32
#define log2_tile_sz 5

#include "GB_cuda_atomics.cuh"

__device__ __inline__ uint64_t GB_cuda_tile_sum_uint64
(
    thread_block_tile<tile_sz> tile,
    uint64_t value
)
{

    //--------------------------------------------------------------------------
    // sum value on all threads to a single value
    //--------------------------------------------------------------------------

    #if (tile_sz == 32)
    {
        // this is the typical case
        value += tile.shfl_down (value, 16) ;
        value += tile.shfl_down (value,  8) ;
        value += tile.shfl_down (value,  4) ;
        value += tile.shfl_down (value,  2) ;
        value += tile.shfl_down (value,  1) ;
    }
    #else
    {
        // tile_sz is less than 32 (either 1, 2, 4, 8, or 16)
        #pragma unroll
        for (int offset = tile_sz >> 1 ; offset > 0 ; offset >>= 1)
        {
            value += tile.shfl_down (value, offset) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // Note that only thread 0 will have the full summation of all values in
    // the tile.  To broadcast it to all threads, use the following:

    // value = tile.shfl (value, 0) ;

    return (value) ;
}

#include "GB_cuda_threadblock_sum_uint64.cuh"

__global__ void GB_cuda_select_bitmap_kernel
(
    int8_t *Cb_out,
    uint64_t *cnvals_out,
    GrB_Matrix A,
    const GB_void *thunk
)
{
    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif

    #if ( GB_A_IS_BITMAP )
    const int8_t *__restrict__ Ab = A->b ;
    #endif

    GB_A_NHELD (anz) ;
    int64_t nrows = A->vlen ;

    uint64_t my_keep = 0 ;
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;
    for (int64_t p = tid ; p < anz ; p += nthreads)
    {
        Cb_out [p] = 0 ;
        if (!GBB_A (Ab, p)) { continue; }

        #if ( GB_DEPENDS_ON_I )
        int64_t i = (p % nrows) ;
        #endif

        #if ( GB_DEPENDS_ON_J )
        int64_t j = (p / nrows) ;
        #endif
        
        #if ( GB_DEPENDS_ON_Y )
        GB_Y_TYPE y = * ((GB_Y_TYPE *) thunk) ;
        #endif

        GB_TEST_VALUE_OF_ENTRY (keep, p) ;
        if (keep) 
        {
            my_keep++ ;
            Cb_out [p] = 1 ;    
        } 
    }
    
    // can make this a warp-level synchronization?
    this_thread_block().sync() ;
    // compute cnvals for this block
    // IMPORTANT: every thread in the threadblock must participate in the warp reduction
    // for thread 0 to obtain the right result
    uint64_t block_keep = GB_cuda_threadblock_sum_uint64 (my_keep) ;
    // this can also be a warp-level synchronization?
    // (we only care about the result in warp 0, since that is where thread 0 is)
    this_thread_block().sync() ;

    if (threadIdx.x == 0)
    {
        // thread 0 updates global cnvals with atomics
        GB_cuda_atomic_add (cnvals_out, block_keep) ;
    }
}


extern "C"
{
    GB_JIT_CUDA_KERNEL_SELECT_BITMAP_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_SELECT_BITMAP_PROTO (GB_jit_kernel)
{
    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    GB_cuda_select_bitmap_kernel <<<grid, block, 0, stream>>> (Cb, cnvals, A, ythunk) ;
    return (GrB_SUCCESS) ;
}