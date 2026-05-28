//------------------------------------------------------------------------------
// gbmx_usage: check usage and make sure GrB.init has been called
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is a gbmx_* utility but it calls GrB_* methods.  However, if GrB_init
// fails, it frees any memory it has allocated (such as the JIT hash table).
// Since GrB_init relies on malloc/free, memory failures are properly handled.

#include "gb_interface.h"

//------------------------------------------------------------------------------
// malloc/free for the default arena 0
//------------------------------------------------------------------------------

typedef void * (*malloc_t) (size_t) ;
typedef void   (*free_t) (void *) ;
static malloc_t gb_malloc0 = malloc ;
static free_t   gb_free0 = free ;

void *gb_malloc (size_t n)
{ 
    // allocate memory in arena 0; at least 8 bytes
    return (gb_malloc0 (MAX (n, sizeof (uint64_t)))) ;
}

void gb_free (void **p)
{
    if (p != NULL && *p != NULL)
    { 
        // free the pointer in arena 0 and set the pointer to NULL to indicate
        // it has been freed.
        gb_free0 (*p) ;
        (*p) = NULL ;
    }
}

//------------------------------------------------------------------------------
// gbmx_usage
//------------------------------------------------------------------------------

void gbmx_usage     // check usage and make sure GrB.init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *usage       // error message if usage is not correct
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
        OK (gb_defaults ( )) ;          // no memory allocated; "cannot" fail

        OK (GrB_Global_get_VOID (GrB_GLOBAL, &gb_malloc0, GxB_ARENA_MALLOC)) ;
        OK (GrB_Global_get_VOID (GrB_GLOBAL, &gb_free0, GxB_ARENA_MALLOC)) ;
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

