//------------------------------------------------------------------------------
// GB_memory.h: memory allocation
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MALLOC_MEMORY_H
#define GB_MALLOC_MEMORY_H

//------------------------------------------------------------------------------
// memory management
//------------------------------------------------------------------------------

void *GB_malloc_memory      // pointer to allocated block of memory
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item,    // sizeof each item
    // output
    size_t *size_allocated  // # of bytes actually allocated
) ;

void GB_free_memory         // free memory
(
    // input/output
    void **p,               // pointer to allocated block of memory to free
    // input
    size_t size_allocated   // # of bytes actually allocated
) ;

#endif

