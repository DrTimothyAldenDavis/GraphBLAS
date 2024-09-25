using namespace cooperative_groups ;

/*
Steps:
1. Build keep array (which entries to keep)
2. Cumsum keep for compression map (Ax -> Cx)
    - Need to know how many entries are preserved
3. Use compression map for building Aj' (preserved cols list)
4. Build change array over compression map
    - Mark where things change
*/

#include "GB_cuda_ek_slice.cuh"
#include "GB_cuda_cumsum.cuh"

// Compute keep array
__global__ void GB_cuda_select_sparse_phase1
(
    int64_t *keep,
    GrB_Matrix A,
    void *ythunk
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
    
    GB_A_NHELD (anz) ;

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    #if ( GB_DEPENDS_ON_J )
    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
            pfirst < anz ;
            pfirst += gridDim.x << log2_chunk_size )
    {
        int64_t my_chunk_size, anvec_sub1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst, chunk_size,
            &kfirst, &klast, &my_chunk_size, &anvec_sub1, &slope) ;

        for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
        {
            int64_t p_final ;
            int64_t k = GB_cuda_ek_slice_entry (&p_final, pdelta, pfirst, Ap, anvec_sub1, kfirst, slope) ;
            int64_t j = GBH_A (Ah, k) ;

            #if ( GB_DEPENDS_ON_I )
            int64_t i = GBI_A (Ai, p_final, A->vlen) ;
            #endif

            GB_TEST_VALUE_OF_ENTRY (p_keep, p_final) ;
            keep[p_final] = p_keep;
            (*n_keep) += p_keep;
        }
    }
    #else
    for (int p = tid; p < anz; p += nthreads)
    {
        #if ( GB_DEPENDS_ON_I )
        int64_t i = GBI_A (Ai, p_final, A->vlen) ;
        #endif

        GB_TEST_VALUE_OF_ENTRY (p_keep, p_final) ;
        keep[p] = p_keep;
        (*n_keep) += p_keep;
    }
    #endif
}

//------------------------------------------------------------------------------
// select sparse, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel) ;
}

/*
inputs to host function:
*/
GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel)
{
    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;

    // Phase 1: keep [p] = 1 if Ai,Ax [p] is kept, 0 otherwise; then cumsum
    int64_t *keep ;
    size_t keep_size ;
    keep = GB_MALLOC_WORK (A->nvals, int64_t, &keep_size) ;
    // FIXME: check for NULL out of memory ...

    GB_cuda_select_sparse_phase1 <<<grid, block, 0, stream>>>
        (keep, A, ythunk) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // keep = cumsum (keep), in-place
    GB_cuda_cumsum (keep, keep, A->nvals, stream, GB_CUDA_CUMSUM_INCLUSIVE_IN_PLACE) ;
    int64_t *map = keep ;   // keep has been replaced with map

    // Phase 2: Build Ci, Cx, and Ak_keep
    // Phase 3: Build delta array over Ak_keep
    // Cumsum over delta array
    // Phase 4: Build Cp and Ch
    // Done!
    
    return (GrB_SUCCESS) ;
}
