#ifndef GB_CUDA_STREAM_POOL
#define GB_CUDA_STREAM_POOL

#include "GB_cuda.hpp"

void GB_cuda_release_stream (int device, cudaStream_t *stream) ;
GrB_Info GB_cuda_grab_stream (int device, cudaStream_t *stream) ;
GrB_Info GB_cuda_stream_pool_init (int ngpus) ;
GrB_Info GB_cuda_stream_pool_finalize () ;

#endif