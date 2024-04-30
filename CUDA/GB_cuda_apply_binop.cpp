#include "GB_cuda_apply.hpp"

#undef GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE ;

#undef GB_FREE_ALL
#define GB_FREE_ALL ;

#define BLOCK_SIZE 512
#define LOG2_BLOCK_SIZE 9

GrB_Info GB_cuda_apply_binop
(
    GB_void *Cx,
    const GrB_Type ctype,
    const GrB_BinaryOp op,
    const GrB_Matrix A,
    const GB_void *scalarx,
    const bool bind1st
)
{

    // FIXME: use the stream pool
    cudaStream_t stream ;
    CUDA_OK (cudaStreamCreate (&stream)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    int32_t gridsz = GB_ICEIL (anz, BLOCK_SIZE) ;
    
    printf ("passed here, is bind1st? %d\n", bind1st) ;
    GrB_Info info ;
    if (bind1st) {
        info = GB_cuda_apply_bind1st_jit (Cx, ctype, op, A, 
            scalarx, stream, gridsz, BLOCK_SIZE) ;
    } else {
        info = GB_cuda_apply_bind2nd_jit (Cx, ctype, op, A,
            scalarx, stream, gridsz, BLOCK_SIZE) ;
    }

    if (info == GrB_NO_VALUE) info = GrB_PANIC ;
    printf ("info is: %d\n", info) ;
    GB_OK (info) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;
    CUDA_OK (cudaStreamDestroy (stream)) ;

    GB_FREE_WORKSPACE ;
    return GrB_SUCCESS ; 

}
