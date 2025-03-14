#include "GB_cuda_stream_pool.hpp"

struct GB_cuda_stream_pool
{
    std::mutex lock ;
    std::condition_variable avail_streams ;
    std::vector<cudaStream_t> streams ;
} ;

static GB_cuda_stream_pool pool ;

#undef GB_FREE_ALL
#define GB_FREE_ALL                                     \
{                                                       \
    while (pool.streams.size ())                        \
    {                                                   \
        cudaStream_t stream = pool.streams.back () ;    \
        pool.streams.pop_back () ;                      \
        cudaStreamDestroy (stream) ;                    \
    }                                                   \
}

void GB_cuda_release_stream (cudaStream_t *stream)
{
    // std::unique_lock uses RAII semantics; it locks the underlying
    // mutex on declaration and unlocks when out-of-scope
    std::unique_lock lock (pool.lock) ;
    pool.streams.push_back (*stream) ;
    pool.avail_streams.notify_one () ;

    (*stream) = nullptr ;
}

void GB_cuda_grab_stream (cudaStream_t *stream)
{
    std::unique_lock lock (pool.lock) ;
    // wait for a stream
    while (!pool.streams.size ())
    {
        pool.avail_streams.wait (lock) ;
    }
    // stream is now available
    (*stream) = pool.streams.back () ;
    pool.streams.pop_back () ;
}

GrB_Info GB_cuda_init_stream_pool (int nstreams)
{
    std::unique_lock lock (pool.lock) ;
    for (int i = 0 ; i < nstreams ; i++)
    {
        cudaStream_t tmp ;
        CUDA_OK (cudaStreamCreate (&tmp)) ;
        pool.streams.push_back (tmp) ;
    }

    return GrB_SUCCESS ;
}