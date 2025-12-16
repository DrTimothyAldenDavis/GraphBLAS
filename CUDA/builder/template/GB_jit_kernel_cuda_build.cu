#define GB_FREE_ALL ;

using namespace cooperative_groups ;

__global__ void GB_cuda_builder_kernel
(

)
{

}

extern "C" {
    GB_JIT_CUDA_KERNEL_BUILDER_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_BUILDER_PROTO (GB_jit_kernel) ;
{
    GB_GET_CALLBACKS ;
    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;

    GB_A_NHELD (anz) ;
    if (anz == 0) return (GrB_SUCCESS) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    GB_cuda_apply_unop_kernel <<<grid, block, 0, stream>>> (Cx, ythunk, A) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}
