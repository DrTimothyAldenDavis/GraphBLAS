//------------------------------------------------------------------------------
// GB_calloc_memory: wrapper for calloc
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A wrapper for calloc.  Space is set to zero.

// The first two parameters are the same as the ANSI C11 calloc, except that
// asking to allocate a block of zero size causes a block of size 1 to be
// allocated instead.  This allows the return pointer p to be checked for the
// out-of-memory condition, even when allocating an object of size zero.

#include "GB.h"

//------------------------------------------------------------------------------
// GB_calloc_helper:  use calloc or malloc/memset to allocate space
//------------------------------------------------------------------------------

static inline void *GB_calloc_helper
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item,    // sizeof each item
    // input/output
    size_t *size_allocated, // on input: # of bytes requested
                            // on output: # of bytes actually allocated
    bool malloc_tracking,
    GB_Context Context
)
{
    void *p ;
    if (GB_Global_have_calloc_function ( ))
    {
        // use the calloc function provided when GraphBLAS was initialized 
        p = (void *) GB_Global_calloc_function (nitems, size_of_item) ;
        if (p == NULL)
        {
            // failed
            (*size_allocated) = 0 ;
        }
        else
        { 
            // success
            if (malloc_tracking) GB_Global_nmalloc_increment ( ) ;
        }
    }
    else
    {
        // no calloc function provided: use malloc and memset instead
        p = (void *) GB_malloc_memory (nitems, size_of_item, size_allocated) ;
        if (p != NULL)
        {
            // clear the block of memory with a parallel memset
            GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
            GB_memset (p, 0, *size_allocated, nthreads_max) ;
        }
    }
    return (p) ;
}

//------------------------------------------------------------------------------
// GB_calloc_memory
//------------------------------------------------------------------------------

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
void *GB_calloc_memory      // pointer to allocated block of memory
(
    size_t nitems,          // number of items to allocate
    size_t size_of_item,    // sizeof each item
    // output
    size_t *size_allocated, // # of bytes actually allocated
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (size_allocated != NULL) ;

    void *p ;
    size_t size ;

    // make sure at least one item is allocated
    nitems = GB_IMAX (1, nitems) ;

    // make sure at least one byte is allocated
    size_of_item = GB_IMAX (1, size_of_item) ;

    bool ok = GB_size_t_multiply (&size, nitems, size_of_item) ;
    if (!ok || nitems > GxB_INDEX_MAX || size_of_item > GxB_INDEX_MAX)
    { 
        // overflow
        (*size_allocated) = 0 ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // allocate the space
    //--------------------------------------------------------------------------

    if (GB_Global_malloc_tracking_get ( ))
    {

        //----------------------------------------------------------------------
        // for memory usage testing only
        //----------------------------------------------------------------------

        // brutal memory debug; pretend to fail if (count-- <= 0).
        bool pretend_to_fail = false ;
        if (GB_Global_malloc_debug_get ( ))
        {
            pretend_to_fail = GB_Global_malloc_debug_count_decrement ( ) ;
        }

        // allocate the memory
        if (pretend_to_fail)
        { 
            p = NULL ;
        }
        else
        { 
            p = (void *) GB_calloc_helper (nitems, size_of_item, &size, true,
                Context) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // normal use, in production
        //----------------------------------------------------------------------

        p = (void *) GB_calloc_helper (nitems, size_of_item, &size, false,
            Context) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*size_allocated) = (p == NULL) ? 0 : size ;
    ASSERT (GB_IMPLIES (p != NULL, size == GB_Global_memtable_size (p))) ;
    return (p) ;
}

