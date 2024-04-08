using namespace cooperative_groups ;

__global__ void GB_cuda_apply_unop_kernel
(
    GB_void *Cx,
    GB_void *thunk_value,
    GrB_Matrix A
)
{

    GB_A_NHELD (nvals) ;
    GB_Y_TYPE thunk ; // uninitialized if no thunk, not used by macro

    #if defined ( GB_DEPENDS_ON_Y )
        // get thunk value (of type GB_Y_TYPE)
        thunk = * ((GB_Y_TYPE *) thunk_value) ;
    #endif

    #if defined ( GB_DEPENDS_ON_J )
        // need to do ek_slice method
    #else
        // can do normal method
        int tid = blockDim.x * blockIdx.x + threadIdx.x ;
        int nthreads = blockDim.x * gridDim.x ;

        for (int p = tid ; p < nvals ; p += nthreads)
        {

        }
    #endif
}

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel)
{
    return (GrB_SUCCESS) ;
}