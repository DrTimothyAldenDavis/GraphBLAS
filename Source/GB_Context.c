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
    // OpenMP on any platform: this is preferred if OpenMP is available
    GxB_Context GB_CONTEXT_THREAD = NULL ;
    #pragma omp threadprivate (GB_CONTEXT_THREAD)
#elif (defined (_WIN64) || defined (_WIN32))
    // Windows: any compiler should support __declspec
    __declspec (thread) GxB_Context GB_CONTEXT_THREAD = NULL ;
#else
    // assume the compiler supports the __thread keyword
    // FIXME: use cmake to see if the __thread keyword is not available,
    // if not, use pthreads
    __thread GxB_Context GB_CONTEXT_THREAD = NULL ;
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
// Context->nthreads_max: # of OpenMP threads to use
//------------------------------------------------------------------------------

//  GB_Context_nthreads_max_get: get max # of threads from a Context
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

//  GB_Context_nthreads_max: get max # of threads from the current Context
int GB_Context_nthreads_max (void)
{ 
    return (GB_Context_nthreads_max_get (GB_CONTEXT_THREAD)) ;
}

//   GB_Context_nthreads_max_set: set max # of threads in a Context
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
// Context->chunk: controls # of threads used for small problems
//------------------------------------------------------------------------------

//     GB_Context_chunk_get: get chunk from a Context
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

//     GB_Context_chunk: get chunk from the Context of this user thread
double GB_Context_chunk (void)
{ 
    return (GB_Context_chunk_get (GB_CONTEXT_THREAD)) ;
}

//   GB_Context_chunk_set: set max # of threads in a Context
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

//------------------------------------------------------------------------------
// Context->gpu_id: which GPU to use
//------------------------------------------------------------------------------

//  GB_Context_gpu_id_get: get max # of threads from a Context
int GB_Context_gpu_id_get (GxB_Context Context)
{
    int gpu_id ;
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_READ
        gpu_id = GxB_CONTEXT_WORLD->gpu_id ;
    }
    else
    { 
        gpu_id = Context->gpu_id ;
    }
    return (gpu_id) ;
}

//  GB_Context_gpu_id: get gpu_id from the current Context
int GB_Context_gpu_id (void)
{ 
    return (GB_Context_gpu_id_get (GB_CONTEXT_THREAD)) ;
}

//   GB_Context_gpu_id_set: set gpu_id in a Context
void GB_Context_gpu_id_set
(
    GxB_Context Context,
    int gpu_id
)
{
    // if gpu_id < 0 or >= # of GPUs in the system: do not use any GPU
    if (Context == NULL || Context == GxB_CONTEXT_WORLD)
    { 
        GB_ATOMIC_WRITE
        GxB_CONTEXT_WORLD->gpu_id = gpu_id ;
    }
    else
    { 
        Context->gpu_id = gpu_id ;
    }
}
