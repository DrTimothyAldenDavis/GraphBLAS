//------------------------------------------------------------------------------
// GB_memory.h: memory allocation
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MEMORY_H
#define GB_MEMORY_H

//------------------------------------------------------------------------------
// memory management
//------------------------------------------------------------------------------

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void *GB_calloc_memory      // pointer to allocated block of memory
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item,    // sizeof each item
    bool unlimited,         // if true, ignore free_pool limits
    // output
    size_t *size_allocated, // # of bytes actually allocated
    GB_Context Context
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void *GB_malloc_memory      // pointer to allocated block of memory
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item,    // sizeof each item
    bool unlimited,         // if true, ignore free_pool limits
    // output
    size_t *size_allocated  // # of bytes actually allocated
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void *GB_realloc_memory     // pointer to reallocated block of memory, or
                            // to original block if the realloc failed.
(
    size_t nitems_new,      // new number of items in the object
    size_t nitems_old,      // old number of items in the object
    size_t size_of_item,    // sizeof each item
    bool unlimited,         // if true, ignore free_pool limits
    // input/output
    void *p,                // old object to reallocate
    // output
    size_t *size_allocated, // # of bytes actually allocated
    bool *ok,               // true if successful, false otherwise
    GB_Context Context
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void GB_free_memory         // free memory, bypassing the free_pool
(
    // input/output
    void **p,               // pointer to allocated block of memory to free
    // input
    size_t size_allocated   // # of bytes actually allocated
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void GB_dealloc_memory      // free memory, return to free_pool or free it
(
    // input/output
    void **p,               // pointer to allocated block of memory to free
    // input
    bool unlimited,         // if true, ignore free_pool limits
    int which_pool,         // 0: malloc pool, 1: calloc pool
    size_t size_allocated   // # of bytes actually allocated
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void GB_free_pool_finalize (void) ;

//------------------------------------------------------------------------------
// malloc/calloc/realloc/free: for permanent contents of GrB objects
//------------------------------------------------------------------------------

// The free_pool limits are followed, so the unlimited parameter is false.
// Block are returned to the malloc pool (which_pool == 0)

#define GB_FREE(p,s) \
    GB_dealloc_memory (p, false, 0, s)

#define GB_CALLOC(n,type,s) \
    (type *) GB_calloc_memory (n, sizeof (type), false, s, Context)

#define GB_MALLOC(n,type,s) \
    (type *) GB_malloc_memory (n, sizeof (type), false, s)

#define GB_REALLOC(p,nnew,nold,type,s,ok,Context) \
    p = (type *) GB_realloc_memory (nnew, nold, sizeof (type), false, \
        (void *) p, s, ok, Context)

//------------------------------------------------------------------------------
// malloc/calloc/realloc/free: for workspace, following free_pool limits
//------------------------------------------------------------------------------

#define GB_CALLOC_WERK(n,type,s) GB_CALLOC(n,type,s)
#define GB_MALLOC_WERK(n,type,s) GB_MALLOC(n,type,s)
#define GB_REALLOC_WERK(p,nnew,nold,type,s,ok,Context) \
             GB_REALLOC(p,nnew,nold,type,s,ok,Context) 
#define GB_FREE_WERK(p,s) GB_FREE(p,s)

//------------------------------------------------------------------------------
// malloc/calloc/realloc/free: for workspace, ignoring free_pool limits
//------------------------------------------------------------------------------

#define GB_FREE_WERK_UNLIMITED_FROM_CALLOC(p,s) \
    GB_dealloc_memory (p, true, 1, s)

#define GB_FREE_WERK_UNLIMITED_FROM_MALLOC(p,s) \
    GB_dealloc_memory (p, true, 0, s)

#define GB_CALLOC_WERK_UNLIMITED(n,type,s) \
    (type *) GB_calloc_memory (n, sizeof (type), true, s, Context)

#define GB_MALLOC_WERK_UNLIMITED(n,type,s) \
    (type *) GB_malloc_memory (n, sizeof (type), true, s)

#endif

