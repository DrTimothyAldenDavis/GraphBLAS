//------------------------------------------------------------------------------
// GB_memcpy: parallel memcpy on the CPU (not for CUDA)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This function uses its own hard-coded chunk size.
// The work is always done on the CPU; CUDA requires its own memcpy method.

#include "GB.h"

#define GB_MEM_CHUNK (1024*1024)

void GB_memcpy                  // parallel memcpy
(
    void *dest,                 // destination
    const int dest_arena,       // destination arena, or -1 if not known
    const void *src,            // source
    const int src_arena,        // src arena, or -1 if not known
    size_t n,                   // # of bytes to copy
    int nthreads                // max # of threads to use (for CPU)
)
{

    //--------------------------------------------------------------------------
    // cudaMemcpy for GPU arenas
    //--------------------------------------------------------------------------

    #if defined ( GRAPHBLAS_HAS_CUDA )
    if (dest_arena >= GxB_NARENAS && src_arena > 0)
    {
        // The src arena is known, and the destination arena is a GPU;
        // use GB_cuda_memcpy.  If it fails, fall back to the CPU memcpy below.
        GrB_Info info = GB_cuda_memcpy (dest, dest_arena, src, src_arena, n,
            (GB_Operator) GrB_IDENTITY_UINT8) ;
        if (info == GrB_SUCCESS)
        {
            // GB_cuda_memcpy was successful
            return ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // parallel memcpy on the CPU(s)
    //--------------------------------------------------------------------------

    if (nthreads <= 1 || n <= GB_MEM_CHUNK)
    { 

        //----------------------------------------------------------------------
        // memcpy using a single thread
        //----------------------------------------------------------------------

        memcpy (dest, src, n) ;
    }
    else
    {

        //----------------------------------------------------------------------
        // memcpy using multiple threads
        //----------------------------------------------------------------------

        size_t nchunks = 1 + (n / GB_MEM_CHUNK) ;
        if (((size_t) nthreads) > nchunks)
        { 
            nthreads = (int) nchunks ;
        }
        GB_void *pdest = (GB_void *) dest ;
        const GB_void *psrc = (GB_void *) src ;

        int64_t k ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (k = 0 ; k < nchunks ; k++)
        {
            size_t start = k * GB_MEM_CHUNK ;
            if (start < n)
            { 
                size_t chunk = GB_IMIN (n - start, GB_MEM_CHUNK) ;
                memcpy (pdest + start, psrc + start, chunk) ;
            }
        }
    }
}

