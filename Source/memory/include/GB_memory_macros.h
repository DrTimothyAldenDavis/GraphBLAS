//------------------------------------------------------------------------------
// GB_memory_macros.h: memory allocation macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MEMORY_MACROS_H
#define GB_MEMORY_MACROS_H

//------------------------------------------------------------------------------
// memory lanes
//------------------------------------------------------------------------------

// The 8-byte memlane_and_memsize (p_mem for short when refering to an object
// p) of a malloc'd object p contains the memlane in the high order byte, and
// the memsize in the lower 7 bytes.

GB_STATIC_INLINE int GB_memlane (uint64_t memlane_and_memsize)
{
    // return the high order byte, containing the memlane
    int memlane = (memlane_and_memsize >> 56) ;
    return (memlane) ;
}

GB_STATIC_INLINE uint64_t GB_memsize (uint64_t memlane_and_memsize)
{
    // return the 7 low order bytes, containing the memsize
    uint64_t memsize = memlane_and_memsize & ((uint64_t) 0x00ffffffffffffffL) ;
    return (memsize) ;
}

GB_STATIC_INLINE uint64_t GB_memlane_and_memsize (int memlane, uint64_t memsize)
{
    // combine the memlane and memsize into the _mem state
    uint64_t memlane_and_memsize = ((uint64_t) memlane) << 56 | memsize ;
    return (memlane_and_memsize) ;
}

GB_STATIC_INLINE uint64_t GB_memlane_change (int memlane, uint64_t memlane_and_size)
{
    // change the memlane of an object, keeping the memsize the same,
    // and return the new memlane_and_memsize state
    uint64_t memsize = GB_memsize (memlane_and_size) ;
    return (GB_memlane_and_memsize (memlane, memsize)) ;
}

//------------------------------------------------------------------------------
// malloc/calloc/realloc/free: for permanent contents of GraphBLAS objects
//------------------------------------------------------------------------------

#ifdef GB_MEMDUMP

    #define GBMDUMP(...) GBDUMP (__VA_ARGS__)

    #define GB_FREE_MEMORY(p,s)                                             \
    {                                                                       \
        if (p != NULL && (*(p)) != NULL)                                    \
        {                                                                   \
            GBMDUMP ("free    %p %8ld: (%s, line %d)\n",                    \
                (void *) (*p), s, __FILE__, __LINE__) ;                     \
        }                                                                   \
        GB_free_memory ((void **) p, s) ;                                   \
    }

    #define GB_MALLOC_MEMORY(n,sizeof_type,s)                               \
        GB_malloc_memory (n, sizeof_type, s) ;                              \
        GBMDUMP ("did malloc: (%s, line %d)\n", __FILE__, __LINE__)

    #define GB_CALLOC_MEMORY(n,sizeof_type,s)                               \
        GB_calloc_memory (n, sizeof_type, s) ;                              \
        GBMDUMP ("did calloc: (%s, line %d)\n", __FILE__, __LINE__)

    #define GB_REALLOC_MEMORY(p,nnew,sizeof_type,s,ok)                      \
    {                                                                       \
        p = GB_realloc_memory (nnew, sizeof_type,                           \
            (void *) p, s, ok) ;                                            \
        GBMDUMP ("did realloc (%s, line %d)\n", __FILE__, __LINE__) ;       \
    }

    #define GB_XALLOC_MEMORY(use_calloc,iso,n,sizeof_type,s)                \
        GB_xalloc_memory (use_calloc, iso, n, sizeof_type, s) ;             \
        GBMDUMP ("did xalloc (%s, line %d)\n", __FILE__, __LINE__)

#else

    #define GBMDUMP(...)

    #define GB_FREE_MEMORY(p,s)                                             \
        GB_free_memory ((void **) p, s)

    #define GB_MALLOC_MEMORY(n,sizeof_type,s)                               \
        GB_malloc_memory (n, sizeof_type, s)

    #define GB_CALLOC_MEMORY(n,sizeof_type,s)                               \
        GB_calloc_memory (n, sizeof_type, s)

    #define GB_REALLOC_MEMORY(p,nnew,sizeof_type,s,ok)                      \
    {                                                                       \
        p = GB_realloc_memory (nnew, sizeof_type, (void *) p, s, ok) ;      \
    }

    #define GB_XALLOC_MEMORY(use_calloc,iso,n,sizeof_type,s)                \
        GB_xalloc_memory (use_calloc, iso, n, sizeof_type, s)

#endif

#endif

