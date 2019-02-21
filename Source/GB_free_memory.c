//------------------------------------------------------------------------------
// GB_free_memory: wrapper for free (used via the GB_FREE_MEMORY macro)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A wrapper for free.  If p is NULL on input, it is not freed.

// This function is called via the GB_FREE_MEMORY(p,n,s) macro.

// not parallel: this function does O(1) work and is already thread-safe.

#include "GB.h"

void GB_free_memory
(
    void *p                 // pointer to allocated block of memory to free
    #ifdef GB_MALLOC_TRACKING
    , size_t nitems         // number of items to free
    , size_t size_of_item   // sizeof each item
    #endif
)
{
    if (p != NULL)
    { 

        #ifdef GB_MALLOC_TRACKING
        {
            // at least one item is always allocated
            nitems = GB_IMAX (1, nitems) ;
            int nmalloc = --GB_Global.nmalloc ;
            GB_Global.inuse -= nitems * size_of_item ;
            #ifdef GB_PRINT_MALLOC
            printf ("Free:    %14p %3d %1d n "GBd" size "GBd"\n",
                p, nmalloc, GB_Global.malloc_debug,
                (int64_t) nitems, (int64_t) size_of_item) ;
            if (nmalloc < 0)
            {
                printf ("%d free    %p negative mallocs!\n", nmalloc, p) ;
            }
            #endif
            ASSERT (nmalloc >= 0) ;
        }
        #endif

        GB_Global.free_function (p) ;
    }
}

