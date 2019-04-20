//------------------------------------------------------------------------------
// GB_realloc_memory: wrapper for realloc_function
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A wrapper for realloc_function.

// This function is called via the GB_REALLOC_MEMORY macro.

// If p is non-NULL on input, it points to a previously allocated object of
// size nitems_old * size_of_item.  The object is reallocated to be of size
// nitems_new * size_of_item.  If p is NULL on input, then a new object of that
// size is allocated.  On success, a pointer to the new object is returned, and
// ok is returned as true.  If the allocation fails, ok is set to false and a
// pointer to the old (unmodified) object is returned.

// Usage:

//      p = GB_realloc_memory (nnew, nold, size, p, &ok)

//      if (ok)

//          p points to a space of size at least nnew*size, and the first
//          part, of size min(nnew,nold)*size, has the same content as
//          the old memory space if it was present.

//      else

//          p points to the old space of size nold*size, which is left
//          unchanged.  This case never occurs if nnew < nold.

#include "GB.h"

void *GB_realloc_memory     // pointer to reallocated block of memory, or
                            // to original block if the reallocation failed.
(
    size_t nitems_new,      // new number of items in the object
    size_t nitems_old,      // old number of items in the object
    size_t size_of_item,    // sizeof each item
    void *p,                // old object to reallocate
    bool *ok                // true if successful, false otherwise
)
{

    size_t size ;

    // make sure at least one item is allocated
    nitems_old = GB_IMAX (1, nitems_old) ;
    nitems_new = GB_IMAX (1, nitems_new) ;

    // make sure at least one byte is allocated
    size_of_item = GB_IMAX (1, size_of_item) ;

    (*ok) = GB_size_t_multiply (&size, nitems_new, size_of_item) ;
    if (!(*ok) || nitems_new > GB_INDEX_MAX || size_of_item > GB_INDEX_MAX)
    { 
        // overflow
        (*ok) = false ;
    }
    else if (p == NULL)
    { 
        // a fresh object is being allocated
        GB_MALLOC_MEMORY (p, nitems_new, size_of_item) ;
        (*ok) = (p != NULL) ;
    }
    else if (nitems_old == nitems_new)
    { 
        // the object does not change; do nothing
        (*ok) = true ;
    }
    else
    { 
        // change the size of the object from nitems_old to nitems_new
        void *pnew ;
        
        //----------------------------------------------------------------------
        // for memory usage testing only
        //----------------------------------------------------------------------

        bool malloc_tracking = GB_Global_malloc_tracking_get ( ) ;
        bool pretend_to_fail = false ;
        bool malloc_debug = false ;
        int nmalloc = 0 ;

        if (malloc_tracking)
        {
            // brutal memory debug; pretend to fail if (count-- <= 0)
            #define GB_CRITICAL_SECTION                                      \
            {                                                                \
                malloc_debug = GB_Global_malloc_debug_get ( ) ;              \
                nmalloc = GB_Global_nmalloc_get ( ) ;                        \
                if (malloc_debug)                                            \
                {                                                            \
                    pretend_to_fail =                                        \
                        GB_Global_malloc_debug_count_decrement ( ) ;         \
                }                                                            \
            }
            bool ok ;
            #include "GB_critical_section.c"
        }

        //----------------------------------------------------------------------
        // reallocate the memory
        //----------------------------------------------------------------------

        if (pretend_to_fail)
        {
            #ifdef GB_PRINT_MALLOC
            printf ("pretend to fail\n") ;
            #endif
            pnew = NULL ;
        }
        else
        {
            // reallocate the space
            pnew = (void *) GB_Global_realloc_function (p, size) ;
        }

        //----------------------------------------------------------------------
        // check if successful
        //----------------------------------------------------------------------

        if (pnew == NULL)
        {
            if (nitems_new < nitems_old)
            {
                // the attempt to reduce the size of the block failed, but
                // the old block is unchanged.  So pretend to succeed.
                (*ok) = true ;
                if (malloc_tracking)
                {
                    // reduce the amount of memory in use
                    #undef  GB_CRITICAL_SECTION
                    #define GB_CRITICAL_SECTION                              \
                    {                                                        \
                        GB_Global_inuse_decrement ((nitems_old - nitems_new) \
                            * size_of_item) ;                                \
                    }
                    bool ok ;
                    #include "GB_critical_section.c"
                }
            }
            else
            {
                // out of memory
                (*ok) = false ;
            }
        }
        else
        {
            // success
            p = pnew ;
            (*ok) = true ;
            if (malloc_tracking)
            {
                bool ok ;
                if (nitems_new < nitems_old)
                {
                    // decrease the amount of memory in use
                    #undef  GB_CRITICAL_SECTION
                    #define GB_CRITICAL_SECTION                              \
                    {                                                        \
                        GB_Global_inuse_decrement ((nitems_old - nitems_new) \
                            * size_of_item) ;                                \
                    }
                    #include "GB_critical_section.c"
                }
                else
                {
                    // increase the amount of memory in use
                    #undef  GB_CRITICAL_SECTION
                    #define GB_CRITICAL_SECTION                              \
                    {                                                        \
                        GB_Global_inuse_increment ((nitems_new - nitems_old) \
                            * size_of_item) ;                                \
                    }
                    #include "GB_critical_section.c"
                }
            }
        }

        //----------------------------------------------------------------------
        // print status
        //----------------------------------------------------------------------

        #ifdef GB_PRINT_MALLOC
        if (malloc_tracking)
        {
            printf ("Realloc: %14p "GBd" %1d n "GBd" -> "GBd" size "GBd"\n",
                pnew, nmalloc, malloc_debug, (int64_t) nitems_old,
                (int64_t) nitems_new, (int64_t) size_of_item) ;
        }
        #endif

    }
    return (p) ;
}

