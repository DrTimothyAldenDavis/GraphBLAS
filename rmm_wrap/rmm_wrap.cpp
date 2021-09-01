
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

//alloc_map is an unordered_map of allocation address to size of each allocation

alloc_map*  get_current_alloc_map() 
{ 
    if (current_handle == NULL)
       return alloc_map();
    else
       return current_handle->ptr_to_size_map.get();
}

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
    std::shared_ptr<alloc_map>                         ptr_to_size_map;
};

RMM_Handle *current_handle = NULL ;

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
    // create size map to lookup size of each allocation 
    handle->ptr_to_size_map = std::shared_ptr< alloc_map> ( alloc_map() );

    current_handle = handle;

}

/*
    GrB_init (mode) ;       // ANSI C11 malloc/calloc/realloc/free, no PMR
    GxB_init (mode, mymalloc, mycalloc, myrealloc, myfree)

    GxB_init (mode, mymalloc, NULL, NULL, myfree)

    GxB_init (mode, mxMalloc, NULL, NULL, mxFree)
    GxB_init (mode, pymalloc, pycalloc, pyrealloc, pyfree)
    GxB_init (mode, jl_malloc, jl_calloc, jl_realloc, jl_free)
    GxB_init (mode, RedisModule_malloc, RedisModule_calloc, RedisModule_realloc, RedisModule_realloc)

    GxB_init (mode, rmm_malloc, rmm_calloc, rmm_realloc, rmm_free)

    void *GxB_Memory_malloc (size_t ) ;
    void  GxB_Memory_free (void *) ;
    void *GxB_Memory_calloc (size_t, size_t) ;      // OK if no calloc
    void *GxB_Memory_realloc (void *, size_t) ;     // NULL if no realloc
*/
#include "pmr_malloc.h"

//------------------------------------------------------------------------------
// rmm_malloc
//------------------------------------------------------------------------------

void *rmm_malloc (std::size_t size)
{
    return (rmm_allocate (&size)) ;
}

//------------------------------------------------------------------------------
// rmm_calloc
//------------------------------------------------------------------------------

void *rmm_calloc (std::size_t n, std::size_t size)
{
    std::size_t s = n * size ;
    void *p = rmm_allocate (&s) ;
    // NOTE: single-threaded on the CPU.  If you want
    // a faster method, malloc the space and use cudaMemset
    // for the GPU or GB_memset on the CPU.
    memset (p, 0, s) ;
    return (p) ;
}

//------------------------------------------------------------------------------
// rmm_realloc
//------------------------------------------------------------------------------

void *rmm_realloc (void *p, std::size_t newsize)
{
    if (p == NULL)
    {
        // allocate a new block.  This is OK.
        return (rmm_allocate (&newsize)) ;
    }

    if (newsize == 0)
    {
        // free the block.  This OK.
        rmm_deallocate (p, 0) ;
        return (NULL) ;
    }

    alloc_map *am = get_current_alloc_map();
    std::size_t oldsize = am->at( (std::size_t)(p) ) ;

    if (oldsize == 0)
    {
        // the block is not in the hashmap; cannot realloc it.
        // This is a failure.
        return (NULL) ;
    }

    // check for quick return
    if (newsize >= oldsize/2 && newsize <= oldsize)
    { 
        // Be lazy. If the block does not change, or is shrinking but only by a
        // small amount, then leave the block as-is.
        return (p) ;
    }

    // allocate the new space
    void *pnew = rmm_allocate (&newsize) ;
    if (pnew == NULL)
    {
        // old block is not modified.  This is a failure, but the old block is
        // still in the hashmap.
        return (NULL) ;
    }

    // copy the old space into the new space
    std::size_t s = (oldsize < newsize) ? oldsize : newsize ;
    // NOTE: single-thread CPU, not GPU.
    // FIXME: query the pointer if it's on the GPU.
    memcpy (pnew, p, s) ;

    // free the old space
    rmm_deallocate (p, oldsize) ;

    // return the new space
    return (pnew) ;
}

//------------------------------------------------------------------------------
// rmm_free
//------------------------------------------------------------------------------

void rmm_free (void *p)
{
    rmm_deallocate (p, 0) ;
}

//------------------------------------------------------------------------------
// rmm_allocate
//------------------------------------------------------------------------------

void *rmm_allocate( std::size_t *size)
{
    // ensure size is nonzero
    if (*size == 0) *size = 256 ;
    // round-up the allocation to a multiple of 256
    std::size_t aligned = (*size) % 256 ;
    if (aligned > 0)
    {
        *size += (256 - aligned) ;
    }
    printf(" rmm_alloc %ld bytes\n",*size);
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();
    void *p = mr->allocate( *size );
    if (p == NULL)
    {
        *size = 0 ;
        return (NULL) ;
    }

    // insert p into the hashmap
    alloc_map *am = get_current_alloc_map();
    if (am == NULL)
    {
       std::cout<< "Uh oh, can't allocate before initializing RMM"<< std::endl;
    }
    else
       am->emplace ( p, *size) ;
}

//------------------------------------------------------------------------------
// rmm_allocate
//------------------------------------------------------------------------------

void rmm_deallocate( void *p, std::size_t size)
{
    //printf("dealloc %ld bytes\n", size); 

    // Note: there are 3 PANIC cases below.  The API of rmm_deallocate does not
    // allow an error condition to be returned.  These PANICs could be logged,
    // or they could terminate the program if debug mode enabled, etc.
    // In production, all we can do is ignore the PANIC.

    if (p == NULL)
    {
        // nothing to do; ignore a double-free
        if (size > 0)
        {
            // PANIC!  Why does a NULL pointer have a nonzero size??
        }
        return ;
    }


    // check the size given.  If the input size is zero, then the
    // size is unknown (say rmm_free(p)).  In that case, just trust the
    // hashmap.  Otherwise, double-check to make sure the size is correct.
    alloc_map *am = get_current_alloc_map();
    size_t actual_size = 0; 
    if ( am == NULL)
    {
       std::cout<< "Uh oh, can't deallocate before initializing RMM"<< std::endl;
    }
    else
    {
       actual_size = am->at( (std::size_t)(p) )  ;
    }

    if (actual_size == 0)
    {
        // PANIC!  oops, p is not in the hashmap.  Ignore it.  TODO: could add
        // a printf here, write to a log file, etc.  if debug mode, abort, etc.
        return ;
    }

    if (size > 0 && size != actual_size)
    {
        // PANIC!  oops, invalid old size.  Ignore the input size, and free p
        // anyway.  TODO: could add a printf here, write to a log file, etc.
        // if debug mode, abort, etc.
    }

    // remove p from the hashmap
    am->erase ( (std::size_t)(p) ) ;

    // deallocate the block of memory
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();
    mr->deallocate( p, actual_size );
}

