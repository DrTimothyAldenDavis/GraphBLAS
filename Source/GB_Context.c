//------------------------------------------------------------------------------
// GB_Context.c: Context object for computational resources
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GxB_Context object contains the set of resources that a user thread
// can use.  There are two kinds of Contexts:

// GxB_CONTEXT_WORLD:  this Context always exists and its contents are always
// defined.  If a user thread has no context, it uses this Context.  It is
// user-visible since its contents may be changed/read by the user application,
// via GxB_Context_set/get.

// GB_CONTEXT_THREAD:  this context is thread-private to each user thread, and
// only visible within this file.  It is not directly accessible by any user
// application.  It is not even visible to other functions inside SuiteSparse:
// GraphBLAS.  If the user thread has not engaged any Context, then
// GB_CONTEXT_THREAD is NULL.

#include "GB.h"

#if defined ( _OPENMP )
    GxB_Context GB_CONTEXT_THREAD = NULL ;
    #pragma omp threadprivate (GB_CONTEXT_THREAD)
#else
    // FIXME: use __declspec(thread) for Windows, __thread for gcc and clang.
    // Otherwise, use Posix pthreads.
    #error "FIXME"
#endif

//------------------------------------------------------------------------------
// GB_Context_engage: engage the Context for a user thread
//------------------------------------------------------------------------------

GrB_Info GB_Context_engage (GxB_Context Context)
{ 
    GB_CONTEXT_THREAD = (Context == GxB_CONTEXT_WORLD) ? NULL : Context ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_Context_disengage: disengage the Context for a user thread
//------------------------------------------------------------------------------

GrB_Info GB_Context_disengage (GxB_Context Context)
{
    if (Context == NULL || Context == GB_CONTEXT_THREAD ||
        Context == GxB_CONTEXT_WORLD)
    { 
        // If no Context provided on input: simply disengage whatever the
        // current Context is for this user thread.  If a non-NULL context is
        // provided, it must match the Context that is currently engaged to
        // this user thread.
        GB_CONTEXT_THREAD = NULL ;
        return (GrB_SUCCESS) ;
    }
    else
    { 
        // A non-NULL Context was provided on input, but it doesn't match the
        // currently engaged Context.
        return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GB_Context_nthreads_max_get: get max # of threads from a Context
//------------------------------------------------------------------------------

int GB_Context_nthreads_max_get (GxB_Context Context)
{
    int nthreads_max ;
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_READ
        nthreads_max = GxB_CONTEXT_WORLD->nthreads_max ;
    }
    else
    { 
        nthreads_max = Context->nthreads_max ;
    }
    return (nthreads_max) ;
}

//------------------------------------------------------------------------------
// GB_Context_nthreads_max: get max # of threads from the current Context
//------------------------------------------------------------------------------

int GB_Context_nthreads_max (void)
{ 
    return (GB_Context_nthreads_max_get (GB_CONTEXT_THREAD)) ;
}

//------------------------------------------------------------------------------
// GB_Context_nthreads_max_set: set max # of threads in a Context
//------------------------------------------------------------------------------

void GB_Context_nthreads_max_set
(
    GxB_Context Context,
    int nthreads_max
)
{
    nthreads_max = GB_IMAX (1, nthreads_max) ;
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_WRITE
        GxB_CONTEXT_WORLD->nthreads_max = nthreads_max ;
    }
    else
    { 
        Context->nthreads_max = nthreads_max ;
    }
}

//------------------------------------------------------------------------------
// GB_Context_chunk_get: get chunk from a Context
//------------------------------------------------------------------------------

double GB_Context_chunk_get (GxB_Context Context)
{
    double chunk ;
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_READ
        chunk = GxB_CONTEXT_WORLD->chunk ;
    }
    else
    { 
        chunk = Context->chunk ;
    }
    return (chunk) ;
}

//------------------------------------------------------------------------------
// GB_Context_chunk: get chunk from the Context of this user thread
//------------------------------------------------------------------------------

double GB_Context_chunk (void)
{ 
    return (GB_Context_chunk_get (GB_CONTEXT_THREAD)) ;
}

//------------------------------------------------------------------------------
// GB_Context_chunk_set: set max # of threads in a Context
//------------------------------------------------------------------------------

void GB_Context_chunk_set
(
    GxB_Context Context,
    double chunk
)
{
    if (chunk < 1) chunk = GB_CHUNK_DEFAULT ;
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_WRITE
        GxB_CONTEXT_WORLD->chunk = chunk ;
    }
    else
    { 
        Context->chunk = chunk ;
    }
}

