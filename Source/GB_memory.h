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
    size_t size_of_item     // sizeof each item
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void *GB_malloc_memory      // pointer to allocated block of memory
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item     // sizeof each item
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void *GB_realloc_memory     // pointer to reallocated block of memory, or
                            // to original block if the realloc failed.
(
    size_t nitems_new,      // new number of items in the object
    size_t nitems_old,      // old number of items in the object
    size_t size_of_item,    // sizeof each item
    void *p,                // old object to reallocate
    bool *ok                // true if successful, false otherwise
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void GB_free_memory
(
    void *p                 // pointer to allocated block of memory to free
) ;

#define GB_FREE(p)                                          \
{                                                           \
    GB_free_memory ((void *) p) ;                           \
    (p) = NULL ;                                            \
}

#define GB_CALLOC(n,type) (type *) GB_calloc_memory (n, sizeof (type))
#define GB_MALLOC(n,type) (type *) GB_malloc_memory (n, sizeof (type))
#define GB_REALLOC(p,nnew,nold,type,ok) \
    p = (type *) GB_realloc_memory (nnew, nold, sizeof (type), (void *) p, ok)

#define GB_CALLOC_WERK(n,type) GB_CALLOC(n,type)
#define GB_MALLOC_WERK(n,type) GB_MALLOC(n,type)
#define GB_REALLOC_WERK(p,nnew,nold,type,ok) GB_REALLOC(p,nnew,nold,type,ok) 
#define GB_FREE_WERK(p) GB_FREE(p)

#endif

