using namespace cooperative_groups ;

// do not #include functions inside of other functions!
#include "GB_cuda_ek_slice.cuh"

#define log2_chunk_size 7
#define chunk_size 128

__global__ void GB_cuda_colscale_kernel
(
    GrB_Matrix C,
    GrB_Matirx A,
    GrB_Matrix D
)
{

    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *__restrict__ Dx = (GB_B_TYPE *) B->x ;
    GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) C->x ;

    #define A_iso GB_A_ISO
    #define D_iso GB_B_ISO

    const int64_t *__restrict__ Ai = A->i ;
    const int64_t *__restrict__ Ap = A->p ;
    GB_A_NVALS (anz) ; // this should be GB_A_NHELD for bitmap/full case
    const int64_t anvec = A->nvec ;

    // TODO: Handle bitmap/full case

    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                 pfirst < anz ;
                 pfirst += gridDim.x << log2_chunk_size )
        {
            int64_t my_chunk_size, anvec_sub1 ;
            float slope ;
            int64_t kfirst = GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst,
                chunk_size, &my_chunk_size, &anvec_sub1, &slope) ;
            
            // alternate:
            // why not just do ek_slice_setup for one thread per block then this_thread_block.sync()?
            // answer:
            // better than having a syncrhonization barrier
            
            // question: why chunks are necessary? why not just do ek_slice_setup across all entries in one go?
            // answer: the slope method is only useful for a small range of entries; non-uniform entry distributions
            //         can distort the usefulness of the slope (will require an exhaustive linear search)
            //         for a large range of entries

            for (int64_t curr_p = threadIdx.x ; curr_p < my_chunk_size ; curr_p += blockDim.x)
            {
                int64_t k = GB_cuda_ek_slice_entry (curr_p, pfirst, Ap, anvec_sub1, kfirst, slope) ;

                GB_DECLAREB (dii) ;
                GB_GETB (dii, Dx, k, D_iso) ;
                GB_DECLAREA (aij) ;
                GB_GETA (aij, Ax, pfirst + curr_p, A_iso) ;
                GB_EWISEOP (Cx, pfirst + curr_p, aij, dii, 0, 0) ;
            }
        }
}

extern "C" {
    GB_JIT_CUDA_KERNEL_COLSCALE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_COLSCALE_PROTO (GB_jit_kernel)
{
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_JUMBLED (D)) ;
    ASSERT (!C->iso) ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    
    GB_cuda_colscale_kernel <<<grid, block, 0, stream>>> (C, A, D) ;

    return (GrB_SUCCESS) ;
}
