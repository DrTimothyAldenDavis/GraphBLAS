#include "GB_cuda_apply.hpp"

#undef GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE ;

#undef GB_FREE_ALL
#define GB_FREE_ALL ;

#define BLOCK_SIZE 512
#define LOG2_BLOCK_SIZE 9

GrB_Info GB_cuda_apply_unop
(
    GB_void *Cx,
    const GrB_Type ctype,
    const GB_Operator op,
    const bool flipij,
    const GrB_Matrix A,
    const GB_void *ythunk
)
{
    // FIXME: use the stream pool
    cudaStream_t stream ;
    CUDA_OK (cudaStreamCreate (&stream)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    int32_t gridsz = 1 + (anz >> LOG2_BLOCK_SIZE) ;

    GrB_Info info = GB_cuda_apply_unop_jit (Cx, ctype, op, flipij, A, 
        ythunk, stream, gridsz, BLOCK_SIZE) ;

    if (info == GrB_NO_VALUE) info = GrB_PANIC ;
    GB_OK (info) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;
    CUDA_OK (cudaStreamDestroy (stream)) ;
    return GrB_SUCCESS ; 

}