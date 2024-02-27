#include "GraphBLAS_cuda.hpp"
#include "GB_cuda.hpp"

#undef GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE ;

#undef GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_cuda_rowscale
(
    GrB_Matrix C,
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    // FIXME: use the stream pool
    // cudaStream_t stream ;
    // CHECK_CUDA (cudaStreamCreate (&stream)) ;

    printf ("Entered GPU rowscale\n") ;

    return GrB_SUCCESS ; 

}