#include "GB_cuda_apply.hpp"

#undef GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE ;

#undef GB_FREE_ALL
#define GB_FREE_ALL ;

#define BLOCK_SIZE 512
#define LOG2_BLOCK_SIZE 9

GrB_Info GB_cuda_apply
(
    GB_void *Cx,
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    // FIXME: use the stream pool
    cudaStream_t stream ;
    CUDA_OK (cudaStreamCreate (&stream)) ;


    CUDA_OK (cudaStreamSynchronize (stream)) ;
    CUDA_OK (cudaStreamDestroy (stream)) ;
    return GrB_SUCCESS ; 

}