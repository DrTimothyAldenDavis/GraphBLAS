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
    // output
    size_t *size_allocated  // # of bytes actually allocated
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void *GB_malloc_memory      // pointer to allocated block of memory
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item,    // sizeof each item
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
    // input/output
    void *p,                // old object to reallocate
    // output
    size_t *size_allocated, // # of bytes actually allocated
    bool *ok                // true if successful, false otherwise
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void GB_free_memory
(
    // input/output
    void **p,               // pointer to allocated block of memory to free
    // input
    size_t size_allocated   // # of bytes actually allocated
) ;

#define GB_FREE(p,s) GB_free_memory (p,s)
#define GB_CALLOC(n,type,s) (type *) GB_calloc_memory (n, sizeof (type), s)
#define GB_MALLOC(n,type,s) (type *) GB_malloc_memory (n, sizeof (type), s)
#define GB_REALLOC(p,nnew,nold,type,s,ok) \
    p = (type *) GB_realloc_memory (nnew, nold, sizeof (type), (void *)p, s, ok)

#define GB_CALLOC_WERK(n,type,s) GB_CALLOC(n,type,s)
#define GB_MALLOC_WERK(n,type,s) GB_MALLOC(n,type,s)
#define GB_REALLOC_WERK(p,nnew,nold,type,s,ok) \
             GB_REALLOC(p,nnew,nold,type,s,ok) 
#define GB_FREE_WERK(p,s) GB_FREE(p,s)

#endif

