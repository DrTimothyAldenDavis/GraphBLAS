#include "GB_cuda_stream_pool.hpp"

struct GB_cuda_stream_pool
{
    std::mutex lock ;
    std::vector<std::condition_variable*> avail_streams ;
    std::vector<std::vector<cudaStream_t>> streams ;
} ;

static GB_cuda_stream_pool pool ;

#undef GB_FREE_ALL
#define GB_FREE_ALL                                                 \
{                                                                   \
    while (device >= 0) {                                           \
        while (pool.streams[device].size ())                        \
        {                                                           \
            cudaStream_t stream = pool.streams[device].back () ;    \
            pool.streams[device].pop_back () ;                      \
            cudaStreamDestroy (stream) ;                            \
        }                                                           \
        delete pool.avail_streams[device] ;                         \
        device-- ;                                                  \
    }                                                               \
}

void GB_cuda_release_stream (int device, cudaStream_t *stream)
{
    if (stream == nullptr)
    {
        return ;
    }
    // std::unique_lock uses RAII semantics; it locks the underlying
    // mutex on declaration and unlocks when out-of-scope
    std::unique_lock lock (pool.lock) ;
    pool.streams[device].push_back (*stream) ;
    pool.avail_streams[device]->notify_one () ;

    (*stream) = nullptr ;
}

void GB_cuda_grab_stream (int device, cudaStream_t *stream)
{
    std::unique_lock lock (pool.lock) ;
    // wait for a stream
    while (!pool.streams[device].size ())
    {
        pool.avail_streams[device]->wait (lock) ;
    }
    // stream is now available
    (*stream) = pool.streams[device].back () ;
    pool.streams[device].pop_back () ;
}

GrB_Info GB_cuda_init_stream_pool (int ngpus, int nstreams_per_gpu)
{
    pool.streams.resize (ngpus) ;
    // std::conditional_variable is not copy-able or assign-able, so
    // need to have a vector of pointers to it to be able to resize
    pool.avail_streams.resize (ngpus) ;
    for (int device = 0 ; device < ngpus ; device++)
    {
        pool.avail_streams [device] = new std::condition_variable () ;
        GB_cuda_set_device (device) ;

        for (int stream = 0 ; stream < nstreams_per_gpu ; stream++)
        {
            cudaStream_t tmp ;
            CUDA_OK (cudaStreamCreate (&tmp)) ;
            pool.streams[device].push_back (tmp) ;
        }
    }

    return GrB_SUCCESS ;
}
