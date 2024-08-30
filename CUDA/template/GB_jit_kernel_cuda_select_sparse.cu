using namespace cooperative_groups ;

/*
select_sparse:
3 phases
Phase 1: column counts
Phase 2: construct Cp
Phase 3: construct Cx, Ci

2 cumsums:
- One for constructing Cp from column counts
- One for construcing Ax -> Cx index mapping
*/

#define tile_sz 32
#define log2_tile_sz 5

#define chunk_size 512
#define log2_chunk_size 9

#define blocksize 64

#include "GB_cuda_atomics.cuh"
#include "GB_cuda_tile_sum_uint64.cuh"
#include "GB_cuda_threadblock_sum_uint64.cuh"

// Compute column counts
__global__ void GB_cuda_select_sparse_phase1
(
    int64_t *col_counts,
    GrB_Matrix A,
    const GB_void *thunk
)
{
    const int64_t *__restrict__ Ap = A->p ;

    #if ( GB_A_IS_HYPER )
    const int64_t *__restrict__ Ah = A->h ;
    #endif

    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif

    #if ( GB_DEPENDS_ON_Y )
    const GB_Y_TYPE y = * ((GB_Y_TYPE *) thunk) ;
    #endif

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    /*
    About the line `int32_t my_col_counts[blocksize]` - see here:
    https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy
    On Ampere (older architecture), there are 64K 32-bit registers per SM.
    Each thread can claim up to 255 registers. Here, we request 64 registers.
    Accounting for other register space used, this means we can conservatively have up to 
    2^9 threads (8 blocks of size 64) running concurrently on a single SM.

    If needed, we can fine tune the block size to balance the overhead of
    ek_slice (smaller blocks mean more binary searches) with SM occupancy
    (larger blocks mean less occupancy due to register usage).
    */
    int32_t my_col_counts[blocksize];

    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
            pfirst < anz ;
            pfirst += gridDim.x << log2_chunk_size )
    {
        int64_t my_chunk_size, anvec_sub1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst, chunk_size,
            &kfirst, &klast, &my_chunk_size, &anvec_sub1, &slope) ;

        // Now, I can update the column counts for col_counts[kfirst:klast]
        for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
        {
            int64_t p_final ;
            int64_t k = GB_cuda_ek_slice_entry (&p_final, pdelta, pfirst, Ap, anvec_sub1, kfirst, slope) ;

            #if ( GB_DEPENDS_ON_J )
            int64_t j = GBH_A (Ah, k) ;
            #endif

            #if ( GB_DEPENDS_ON_I )
            int64_t i = GBI_A (Ai, p_final, A->vlen) ;
            #endif

            GB_TEST_VALUE_OF_ENTRY (keep, p_final) ;
            if (keep) 
            {
                my_col_counts[k - kfirst]++;
            }
        }
        // Due to implicit warp sync, at this point, each warp has
        // its column counts computed. Do the threadblock sums.
        for (int k = 0; k <= klast - kfirst; k++)
        {
            // This might be pretty slow: we are doing (klast - kfirst) block syncs.
            // I'm not sure how we could avoid this, though.
            uint32_t block_sum = GB_cuda_threadblock_sum_uint64 (my_col_counts[k]) ;
            if (threadIdx.x == 0)
            {
                if (k == 0 || k == (klast - kfirst))
                {
                    // atomic add on the edges
                } else {
                    // no need for atomics for intermediate k
                    col_counts[kfirst + k] = block_sum;
                }
            }
        }
    }

}


extern "C"
{
    GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel) ;
}

/*
inputs to host function:
[out] C: output matrix
[in]  A: input matrix
*/
GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel)
{
    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;

    int64_t *col_counts ;
    col_counts = (int64_t *) malloc(A->ncols * sizeof(uint64_t)) ;

    GB_cuda_select_sparse_phase1 <<<grid, block, 0, stream>>> (col_counts, A, ythunk) ;
    return (GrB_SUCCESS) ;
}
