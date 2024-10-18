using namespace cooperative_groups ;

/*
Steps:
1. Build Keep array (which entries to Keep)
2. Cumsum Keep for compression Map (Ax -> Cx)
    - Need to know how many entries are preserved
3. Use compression Map for building Aj' (preserved cols list)
4. Build change array over compression Map
    - Mark where things change
*/

#include "GB_cuda_ek_slice.cuh"
#include "GB_cuda_cumsum.cuh"

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase1: construct Keep array
//------------------------------------------------------------------------------

// Compute Keep array
__global__ void GB_cuda_select_sparse_phase1
(
    int64_t *Keep,
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

    #if ( GB_DEPENDS_ON_J )

        for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                pfirst < anz ;
                pfirst += gridDim.x << log2_chunk_size )
        {
            int64_t my_chunk_size, anvec1, kfirst, klast ;
            float slope ;
            GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst, chunk_size,
                &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

            for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
            {
                int64_t pA ;
                int64_t k = GB_cuda_ek_slice_entry (&pA, pdelta, pfirst, Ap, anvec1, kfirst, slope) ;
                int64_t j = GBH_A (Ah, k) ;

                #if ( GB_DEPENDS_ON_I )
                int64_t i = GBI_A (Ai, pA, A->vlen) ;
                #endif

                GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
                Keep[pA] = keep;
            }
        }

    #else

        int tid = blockIdx.x * blockDim.x + threadIdx.x ;
        int nthreads = blockDim.x * gridDim.x ;

        for (int64_t pA = tid; pA < anz; pA += nthreads)
        {
            #if ( GB_DEPENDS_ON_I )
            int64_t i = GBI_A (Ai, pA, A->vlen) ;
            #endif

            GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
            Keep[pA] = keep;
        }

    #endif
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase2:
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase2
(
    int64_t *Map
    GrB_Matrix A,
    int64_t *Ak_keep,
    int64_t *Ck_delta,
    int64_t *Ci,
    GB_X_TYPE *Cx
)
{
    const int64_t *__restrict__ Ap = A->p ;
    const int64_t *__restrict__ Ai = A->i ;
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    
    GB_A_NHELD (anz) ;
    int64_t cnz = Map[anz - 1];

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
        pfirst < anz ;
        pfirst += gridDim.x << log2_chunk_size )
    {
        int64_t my_chunk_size, anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst, chunk_size,
            &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

        for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
        {
            int64_t pA ;
            int64_t kA = GB_cuda_ek_slice_entry (&pA, pdelta, pfirst, Ap, anvec1, kfirst, slope) ;
            int64_t pC = Map [pA] ;     // Note: this is off-by-1 (see below).
            if (Map [pA-1] < pC)
            {
                // This entry is kept; Keep [pA] was 1 but the contents of the
                // Keep has been overwritten by the Map array using an
                // inclusive cumsum.  Keep [pA] (before being overwritten) is
                // identical to the expression (Map [pA-1] < pC).

                // Map is offset by 1 since it was computed as an inclusive cumsum,
                // so decrement pC here to get the actual position in Ci,Cx.
                pC-- ;
                Ci[pC] = GBI_A (Ai, pA, A->vlen) ;
                Cx[pC] = Ax[pA] ;
                Ak_keep[pC] = kA + 1 ;
            }
        }

        // Build the Delta over Ck_delta
        this_thread_block().sync();

        __shared__ int8_t do_keep[blockDim.x] ;
        
        for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
        {
            int64_t pA = pfirst + pdelta ;
            int64_t pC = Map[pA] ;
            do_keep[pA] = (Ak_keep[pC] != Ak_keep[pC - 1]) ;

        }
        // No sync barrier needed; threads only look at their own entries
        for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
        {
            int64_t pA = pfirst + pdelta ;
            int64_t pC = Map[pA] ;
            Ck_delta[pC] = do_keep[pA] ;
        }
    }
}

__global__ void GB_cuda_select_sparse_phase3
(
    GrB_Matrix A,
    int64_t *cnz_ptr,
    int64_t *Ak_keep,
    int64_t *Ck_map,
    int64_t *Cp,
    int64_t *Ch,
)
{
    const int64_t *__restrict__ Ah = A->h;
    int64_t cnz = *(cnz_ptr) ;

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t pC = tid; pC < cnz; pC += nthreads)
    {
        if (Ck_map[pC] != Ck_map[pC - 1])
        {
            int64_t kA = Ak_keep[pC] - 1 ;
            Cp[Ck_map[pC] - 1] = pC;
            Ch[Ck_map[pC] - 1] = GBH_A (Ah, kA);
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

    GrB_Info ret = GrB_SUCCESS ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;

    // Phase 1: Keep [p] = 1 if Ai,Ax [p] is kept, 0 otherwise; then cumsum
    int64_t *W, *W_2, *Ak_keep, *Ck_delta, *Keep ;
    int64_t *cnz ;
    W = W_2 = Ak_keep = cnz = NULL ;
    size_t W_size, W_2_size, cnz_size, Ak_keep_size ;
    W = GB_MALLOC_WORK (A->nvals + 1, int64_t, &W_size) ;
    cnz = GB_MALLOC_WORK (1, int64_t, &cnz_size);

    if (W == NULL || cnz == NULL) {
        ret = GrB_OUT_OF_MEMORY ;
        goto done;
    }

    W [0] = 0;     // placeholder for easier end-condition
    Keep = W + 1 ;  // Keep has size A->nvals and starts at W [1]

    GB_cuda_select_sparse_phase1 <<<grid, block, 0, stream>>>
        (Keep, A, ythunk) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Keep = cumsum (Keep), in-place
    GB_cuda_cumsum (Keep, Keep, A->nvals, stream, GB_CUDA_CUMSUM_INCLUSIVE) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    (*cnz) = Keep[A->nvals - 1];
    W_2 = GB_MALLOC_WORK ((*cnz) + 1, int64_t, &W_2_size) ;
    Ak_keep = GB_MALLOC_WORK ((*cnz), int64_t, &Ak_keep_size) ;
    if (W_2 == NULL || Ak_keep == NULL) {
        ret = GrB_OUT_OF_MEMORY ;
        goto done;
    }
    W_2[0] = 0 ;
    Ck_delta = W_2 + 1 ;
    int64_t *Map = Keep ;   // Keep has been replaced with Map

    // Phase 2: Build Ci, Cx, Ak_keep, and Ck_delta
    GB_cuda_select_sparse_phase2 <<<grid, block, 0, stream>>>
        (Map, A, Ak_keep, Ck_delta, Ci, Cx) ;
    
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    // Cumsum over Ck_delta array -> Ck_map
    // Can reuse `Keep` to avoid a malloc
    CUDA_OK (GB_cuda_cumsum (Keep, Ck_delta, (*cnz), stream, GB_CUDA_CUMSUM_INCLUSIVE)) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    int64_t *Ck_map = Keep;

    // Phase 3: Build Cp and Ch
    GB_cuda_select_sparse_phase3 <<<grid, block, 0, stream>>>
        (A, cnz, Ak_keep, Ck_map, Cp, Ch) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    
    Cp[Ck_map[cnz - 1]] = cnz;
    Ch[Ck_map[cnz - 1]] = GBH(A->h, Ck_delta[cnz - 1]);
    // Done!
done:
    GB_FREE_WORK (W) ;
    GB_FREE_WORK (W_2) ;
    GB_FREE_WORK (Ak_keep) ;
    GB_FREE_WORK (cnz) ;

    return ret ;
}
