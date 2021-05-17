
//#includePath += "../rmm/include/rmm"
#include <rmm_wrap.hpp>

inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>();}

inline auto make_and_set_pool(size_t initial_size, size_t maximum_size) 
{ 
	auto resource = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>
			        ( make_managed(), initial_size, maximum_size );
	rmm::mr::set_current_device_resource( resource.get());
	return resource;
}

typedef rmm::mr::managed_memory_resource managed_mr;
typedef rmm::mr::pool_memory_resource<managed_mr> managed_pool_mr;

struct RMM_Handle 
{
    std::shared_ptr<rmm::mr::device_memory_resource> resource; 
};

void rmm_create_handle( RMM_Handle *handle)
{ //place a new RMM pool at this handle in memory
    handle = new RMM_Handle(); 
}

void rmm_destroy_handle ( RMM_Handle *handle)
{
    free(handle);
}

void rmm_initialize(RMM_Handle *handle, size_t init_pool_size, size_t max_pool_size)
{
    // Construct a resource that uses a coalescing best-fit pool allocator
    handle->resource =  make_and_set_pool( init_pool_size, max_pool_size) ;
}


void *rmm_allocate( size_t *size)
{
    if (*size % 256 > 0)
    {
        *size = *size + (*size%256);
    }
    printf("rmm_alloc %ld bytes\n",*size);
    rmm::mr::device_memory_resource *mr=  rmm::mr::get_current_device_resource();
    return mr->allocate( *size );
}

void rmm_deallocate( void *p, size_t size)
{
    printf("dealloc %ld bytes\n", size); 
    rmm::mr::device_memory_resource *mr=  rmm::mr::get_current_device_resource();
    mr->deallocate( p, size );
}

