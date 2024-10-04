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
    const int64_t *__restrict__ Ah = A->h ;

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
            int64_t pfinal ;
            int64_t k = GB_cuda_ek_slice_entry (&pfinal, pdelta, pfirst, Ap, anvec_sub1, kfirst, slope) ;
            int64_t j = GBH_A (Ah, k) ;

            #if ( GB_DEPENDS_ON_I )
            int64_t i = GBI_A (Ai, pfinal, A->vlen) ;
            #endif

            GB_TEST_VALUE_OF_ENTRY (p_keep, pfinal) ;
            keep[pfinal] = p_keep;
            (*n_keep) += p_keep;
        }
    }
    #else
    for (int64_t p = tid; p < anz; p += nthreads)
    {
        #if ( GB_DEPENDS_ON_I )
        int64_t i = GBI_A (Ai, p, A->vlen) ;
        #endif

        GB_TEST_VALUE_OF_ENTRY (p_keep, p) ;
        keep[p] = p_keep;
        (*n_keep) += p_keep;
    }
    #endif
}

__global__ void GB_cuda_select_sparse_phase2
(
    int64_t *map
    GrB_Matrix A,
    int64_t *Ak_keep,
    int64_t *Ci,
    GB_X_TYPE *Cx
)
{
    const int64_t *__restrict__ Ap = A->p ;
    const int64_t *__restrict__ Ai = A->i ;
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    
    GB_A_NHELD (anz) ;
    int64_t cnz = map[anz - 1];

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

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
            int64_t pfinal ;
            int64_t k = GB_cuda_ek_slice_entry (&pfinal, pdelta, pfirst, Ap, anvec_sub1, kfirst, slope) ;
            Ci[map[pfinal]] = GBI_A (Ai, pfinal, A->vlen);
            Cx[map[pfinal]] = Ax[pfinal];
            Ak_keep[map[pfinal]] = k;
        }

        // Build the delta over Ak_keep
        this_thread_block().sync();
        
        for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
        {
            int64_t pfinal = pfirst + pdelta;
            Ak_keep[pfinal] = (pfinal != 0) && (Ak_keep[pfinal] != Ak_keep[pfinal - 1]);
        }
    }
}

__global__ void GB_cuda_select_sparse_phase3
(
    GrB_Matrix A,
    void *cnz_ptr,
    int64_t *Ak_keep,
    int64_t *Ak_map,
    int64_t *Cp,
    int64_t *Ch,
)
{
    const int64_t *__restrict__ Ah = A->h;
    int64_t cnz = * ((int64_t *) cnz_ptr);

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t p = tid; p < cnz; p += nthreads)
    {
        if ((p > 0) && (Ak_map[p] != Ak_map[p - 1]))
        {
            Cp[Ak_map[p] - 1] = p;
            Ch[Ak_map[p] - 1] = GBH_A (Ah, Ak_keep[p - 1]);
        }
    }
}

//------------------------------------------------------------------------------
// select sparse, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel)
{
    ASSERT(GB_A_IS_HYPER);

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;

    // Phase 1: keep [p] = 1 if Ai,Ax [p] is kept, 0 otherwise; then cumsum
    int64_t *keep, *Ak_keep;
    int64_t *cnz;
    size_t keep_size, Ak_keep_size, cnz_size;
    keep = GB_MALLOC_WORK (A->nvals, int64_t, &keep_size) ;
    cnz = GB_MALLOC_WORK (1, int64_t, &cnz_size);
    if (keep == NULL || cnz_size == NULL) {
        return GrB_OUT_OF_MEMORY;
    }

    GB_cuda_select_sparse_phase1 <<<grid, block, 0, stream>>>
        (keep, A, ythunk) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // keep = cumsum (keep), in-place
    GB_cuda_cumsum (keep, keep, A->nvals, stream, GB_CUDA_CUMSUM_INCLUSIVE) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    (*cnz) = keep[A->nvals - 1];
    Ak_keep = GB_MALLOC_WORK ((*cnz), int64_t, &Ak_keep_size);
    if (Ak_keep == NULL) {
        return GrB_OUT_OF_MEMORY;
    }

    int64_t *map = keep ;   // keep has been replaced with map

    // Phase 2: Build Ci, Cx, and Ak_keep
    GB_cuda_select_sparse_phase2 <<<grid, block, 0, stream>>>
        (map, A, Ak_keep, Ci, Cx) ;
    
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    // Cumsum over delta array -> Ak_map
    // Can reuse `keep` to avoid a malloc
    GB_cuda_cumsum (keep, Ak_keep, (*cnz), stream, GB_CUDA_CUMSUM_INCLUSIVE) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    int64_t *Ak_map = keep;
    // Phase 3: Build Cp and Ch
    GB_cuda_select_sparse_phase3 <<<grid, block, 0, stream>>>
        (A, (void *) cnz, Ak_keep, Ak_map, Cp, Ch) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    
    Cp[Ak_map[cnz - 1]] = cnz;
    Ch[Ak_map[cnz - 1]] = GBH(A->h, Ak_keep[cnz - 1]);
    // Done!
    
    return (GrB_SUCCESS) ;
}
