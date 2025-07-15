#include "GB_cuda_stream_pool.hpp"

#define STREAMS_PER_DEVICE 32

struct GB_cuda_stream_pool
{
    std::vector<std::array<cudaStream_t, STREAMS_PER_DEVICE>> streams ;
    std::vector<int> nstreams_avail;
} ;

static GB_cuda_stream_pool pool ;

#undef GB_FREE_ALL
#define GB_FREE_ALL                                     \
{                                                       \
    while (pool.streams.size())                         \
    {                                                   \
        auto curr = pool.streams.back() ;               \
        int end = pool.nstreams_avail.back() - 1 ;      \
        for (int idx = end; idx >= 0 ; idx--)           \
        {                                               \
            cudaStreamDestroy (curr[idx]) ;             \
        }                                               \
        pool.streams.pop_back() ;                       \
        pool.nstreams_avail.pop_back() ;                \
    }                                                   \
}

void GB_cuda_release_stream (int device, cudaStream_t *stream)
{
    if (stream == nullptr || (*stream) == nullptr)
    {
        return ;
    }

    ASSERT (device < pool.streams.size()) ;

    GB_OPENMP_LOCK_SET (4)
//  #pragma omp critical
    {
        if (pool.nstreams_avail[device] == STREAMS_PER_DEVICE)
        {
            // Pool is full; destroy the stream
            cudaStreamDestroy (*stream) ;
        }
        else
        {
            // Check the stream back in; it's OK if the stream wasn't
            // created at init time. Whatever stream was will be destroyed
            // when it is checked back in.
            size_t stream_idx = pool.nstreams_avail[device];
            pool.streams[device][stream_idx] = (*stream) ;
            pool.nstreams_avail[device]++ ;
        }
    }
    GB_OPENMP_LOCK_UNSET (4)
    (*stream) = nullptr ;
}

GrB_Info GB_cuda_grab_stream (int device, cudaStream_t *stream)
{
    ASSERT (stream != nullptr) ;
    ASSERT (device < pool.streams.size()) ;
    GrB_Info ret = GrB_SUCCESS ;
    cudaError_t cuda_error1 = cudaSuccess ;

    GB_OPENMP_LOCK_SET (4)
//  #pragma omp critical
    {
        if (!pool.nstreams_avail[device])
        {
            // Pool is empty; create a stream
            GB_cuda_set_device (device) ;
            cuda_error1 = cudaStreamCreate (stream) ;
        }
        else
        {
            // Checkout a stream
            size_t stream_idx = pool.nstreams_avail[device] - 1;
            (*stream) = pool.streams[device][stream_idx];
            pool.nstreams_avail[device]-- ;
        }
    }
    GB_OPENMP_LOCK_UNSET (4)
    CUDA_OK (cuda_error1) ;
    return ret ;
}

GrB_Info GB_cuda_stream_pool_init (int ngpus)
{
    for (int device = 0 ; device < ngpus ; device++)
    {
        pool.nstreams_avail.push_back (0) ;
        pool.streams.push_back (std::array<cudaStream_t, STREAMS_PER_DEVICE>()) ;
        GB_cuda_set_device (device) ;

        for (int stream = 0 ; stream < STREAMS_PER_DEVICE ; stream++)
        {
            cudaStream_t tmp ;
            CUDA_OK (cudaStreamCreate (&tmp)) ;
            pool.streams[device][stream] = tmp ;
            pool.nstreams_avail[device]++ ;
        }
    }

    return GrB_SUCCESS ;
}

GrB_Info GB_cuda_stream_pool_finalize ()
{
    GB_FREE_ALL ;
    return GrB_SUCCESS ;
}
