
//#includePath += "../rmm/include/rmm"
#include "rmm_wrap.hpp"
#include <iostream>


//inline auto make_host() { return std::make_shared<rmm::mr::new_delete_resource>(); }

//inline auto make_host_pinned() { return std::make_shared<rmm::mr::pinned_memory_resource>(); }

inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }


//inline auto make_and_set_host_pool(std::size_t initial_size, std::size_t maximum_size) 
//{ 
//        auto resource = std::pmr::synchronized_pool_resource(); 
//                       
//        rmm::mr::set_current_device_resource( resource );
//        return resource;
//}

 // inline auto make_and_set_host_pinned_pool(std::size_t initial_size, std::size_t maximum_size) 
 // { 
 //         auto resource = rmm::mr::make_owning_wrapper<pool_mr>
 //                                 ( make_host_pinned(), initial_size, maximum_size );
 //         rmm::mr::set_current_device_resource( resource.get());
 //         return resource;
 // }

inline auto make_and_set_device_pool(std::size_t initial_size, std::size_t maximum_size) 
{ 
	auto resource = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>
			        ( make_cuda(), initial_size, maximum_size );
	rmm::mr::set_current_device_resource( resource.get());
	return resource;
}

inline auto make_and_set_managed_pool(std::size_t initial_size, std::size_t maximum_size) 
{ 
    std::cout<< " make_managed_pool called with  init_size "<<initial_size<<" max_size "<<maximum_size<<"\n";
	auto resource = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>
			        ( make_managed(), initial_size, maximum_size );
	rmm::mr::set_current_device_resource( resource.get());
	return resource;
}



struct RMM_Handle 
{
    RMM_MODE mode;
    std::shared_ptr<rmm::mr::device_memory_resource>   resource; 
    std::shared_ptr<std::pmr::memory_resource>         host_resource;
    // add a hash table
};

void rmm_create_handle( RMM_Handle **handle)
{ //place a new RMM pool at this handle in memory
    *handle = new RMM_Handle(); 
}

void rmm_destroy_handle ( RMM_Handle **handle)
{
    delete(*handle);
    *handle = NULL;
}

void rmm_initialize(RMM_Handle *handle, RMM_MODE mode,  std::size_t init_pool_size, std::size_t max_pool_size)
{
    std::cout<< " init called with mode "<<mode<<" init_size "<<init_pool_size<<" max_size "<<max_pool_size<<"\n";
    // Mark down the mode for reference later
    handle->mode = mode;

    // Construct a resource that uses a coalescing best-fit pool allocator
    if (mode == rmm_host )
    {
        //handle->host_resource =  std::pmr::synchronized_pool_resource(); // (init_pool_size, max_pool_size) ;
        //handle->host_resource =  make_and_set_host_pool(); // (init_pool_size, max_pool_size) ;
    }
    else if (mode == rmm_host_pinned )
    {
      //  handle->host_resource =  std::pmr::synchronized_pool_resource(); // (init_pool_size, max_pool_size) ;
    }
    else if (mode == rmm_device )
    {
        handle->resource =  make_and_set_device_pool( init_pool_size, max_pool_size) ;
    }
    else if ( mode == rmm_managed )
    {
        handle->resource =  make_and_set_managed_pool( init_pool_size, max_pool_size) ;
    }
    else
    {
        //TODO, handle other cases or an error here      
    }
}


#if 0
void *rmm_malloc (std::size_t size)
{
    size_t *p = (size_t *) rmm_allocate (&size) ;
    hash_insert (p, size) ;
    return (p) ;
}

void *rmm_calloc (std::size_t n, std::size_t size)
{
    // ...
}

void *rmm_realloc (...std::size_t size)
{
}

void rmm_free (void *p)
{
    size_t size = hash_lookup (p)
    rmm_deallocate (p, size) ;
}
#endif

void *rmm_allocate( std::size_t *size)
{

    std::size_t aligned = (*size) % 256;
    if (aligned > 0)
    {
        *size += (256 -aligned);
    }
    printf(" rmm_alloc %ld bytes\n",*size);
    rmm::mr::device_memory_resource *mr=  rmm::mr::get_current_device_resource();
    return mr->allocate( *size );
}

void rmm_deallocate( void *p, std::size_t size)
{
    //printf("dealloc %ld bytes\n", size); 
    rmm::mr::device_memory_resource *mr=  rmm::mr::get_current_device_resource();
    mr->deallocate( p, size );
}

