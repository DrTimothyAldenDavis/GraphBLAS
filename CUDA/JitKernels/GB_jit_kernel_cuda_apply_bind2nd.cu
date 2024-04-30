using namespace cooperative_groups ;

__global__ void GB_cuda_apply_bind2nd_kernel
(
    GB_void *Cx_out,
    GrB_Matrix A,
    GB_void *scalarx,
)
{
    const GB_X_TYPE x = * ((GB_X_TYPE *) scalarx) ; // gets scalarx [0]
    const GB_A_TYPE *__restrict__ Ax = (GB_B_TYPE *) A->x ;
    GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) Cx_out ;

    const int8_t *__restrict__ Ab = A->b ;

    GB_A_NHELD (nvals) ;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x ;

    for (int p = tid ; p < nvals ; p += nthreads)
    {
        if (!GBB_A (Ab, p)) { continue ; }
        GB_DECLAREA (aij) ;
        GB_GETA (aij, ax, p, false) ;
        GB_EWISEOP (Cx, p, aij, x, /* i */, /* j */) ;
    }
}

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_BIND1ST_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_BIND1ST_PROTO (GB_jit_kernel)
{
    ASSERT (Cx != NULL) ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    
    GB_cuda_apply_bind1st_kernel <<<grid, block, 0, stream>>> (Cx, scalarx, B) ;

    return (GrB_SUCCESS) ;
}
