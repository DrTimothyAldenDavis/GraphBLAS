#include "stddef.h"
#include <cuda.h>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>

struct RMM_Handle;

void rmm_create_handle( RMM_Handle *);
void rmm_destroy_handle( RMM_Handle *);

void rmm_initialize( RMM_Handle *handle, size_t init_pool_size, size_t max_pool_size) ;

void *rmm_allocate(  size_t *size) ;
void rmm_deallocate( void *p, size_t size) ;
