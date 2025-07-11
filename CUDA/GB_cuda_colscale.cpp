#include "GB_cuda_ewise.hpp"

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                   \
{                                                           \
    if (stream != nullptr)                                  \
    {                                                       \
        cudaStreamSynchronize (stream) ;                    \
        GB_cuda_release_stream (device, &stream) ;          \
    }                                                       \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_WORKSPACE

#define BLOCK_SIZE 128
#define LOG2_BLOCK_SIZE 7

GrB_Info GB_cuda_colscale
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    int device ;
    cudaStream_t stream = nullptr ;

    CUDA_OK (cudaGetDevice (&device)) ;
    GB_cuda_grab_stream (device, &stream) ;

    GrB_Info info ;

    // compute gridsz, blocksz, call GB_cuda_rowscale_jit
    GrB_Index anz = GB_nnz_held (A) ;
    
    int32_t gridsz = 1 + (anz >> LOG2_BLOCK_SIZE) ;

    GB_OK (GB_cuda_colscale_jit ( C, A, D, 
        semiring->multiply, flipxy, stream, gridsz, BLOCK_SIZE)) ;
    
    GB_FREE_WORKSPACE ;
    return GrB_SUCCESS ; 

}
