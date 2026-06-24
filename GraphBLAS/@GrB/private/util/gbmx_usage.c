//------------------------------------------------------------------------------
// gbmx_usage: check usage and make sure GrB.init has been called
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is a gbmx_* utility but it calls GrB_* methods.  However, if GrB_init
// fails, it frees any memory it has allocated (such as the JIT hash table).
// Since GrB_init relies on the default allocates in arena 0 (malloc/free),
// memory failures are properly handled.

#include "gb_interface.h"

//------------------------------------------------------------------------------
// malloc/free for each arena
//------------------------------------------------------------------------------

typedef void * (*malloc_t) (size_t) ;
typedef void   (*free_t) (void *) ;
static malloc_t gb_malloc_func [4] = { malloc, NULL, mxMalloc, NULL } ;
static free_t   gb_free_func   [4] = { free  , NULL, mxFree  , NULL } ;

void *gb_malloc (size_t n, int arena)
{ 
    // allocate memory in the arena; at least 8 bytes
    if (arena < 0 || arena >= 4 || gb_malloc_func [arena] == NULL)
    {
        return (NULL) ;
    }
    return (gb_malloc_func [arena] (MAX (n, sizeof (uint64_t)))) ;
}

void gb_free (void **p, int arena)
{
    if (p != NULL && *p != NULL && arena >= 0 && arena < 4
        && gb_free_func [arena] != NULL)
    { 
        // free the pointer in the arena and set the pointer to NULL to indicate
        // it has been freed.
        gb_free_func [arena] (*p) ;
        (*p) = NULL ;
    }
}

//------------------------------------------------------------------------------
// gbmx_usage
//------------------------------------------------------------------------------

void gbmx_usage     // check usage and make sure GrB.init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *usage,      // error message if usage is not correct
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // make sure GrB.init has been called
    //--------------------------------------------------------------------------

    int GrB_init_has_been_called = 0 ;
    GxB_initialized (&GrB_init_has_been_called) ;

    if (!GrB_init_has_been_called)
    {

        //----------------------------------------------------------------------
        // tell MATLAB to call GrB_finalize when this mexFunction is cleared
        //----------------------------------------------------------------------

        mexAtExit (gb_at_exit) ;

        //----------------------------------------------------------------------
        // initialize GraphBLAS and set defaults for its use in MATLAB
        //----------------------------------------------------------------------

        OK (GrB_init (GrB_NONBLOCKING)) ;

        // use mxMalloc/mxFree for the MATLAB arena
        OK (GxB_arena_init (MXARENA, mxMalloc, mxCalloc, mxRealloc, mxFree)) ;

        OK (gb_defaults (err)) ;        // no memory allocated; "cannot" fail

        // acquire malloc/free of each arena for gb_malloc and gb_free
        for (int arena = 0 ; arena < 4 ; arena++)
        {
            OK (GrB_Global_get_VOID (GrB_GLOBAL, &(gb_malloc_func [arena]),
                GxB_ARENA_MALLOC)) ;
            OK (GrB_Global_get_VOID (GrB_GLOBAL, &(gb_free_func [arena]),
                GxB_ARENA_FREE)) ;
        }
    }

    //--------------------------------------------------------------------------
    // check usage
    //--------------------------------------------------------------------------

    if (!ok)
    {
        ERROR (usage, GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // get test coverage (not used in production; for testing only)
    //--------------------------------------------------------------------------

    gbcov_get ( ) ;
}

