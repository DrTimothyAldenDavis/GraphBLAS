//------------------------------------------------------------------------------
// GraphBLAS/CUDA/Template/GB_cuda_ek_slice.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_cuda_ek_slice finds the vector k that owns each entry in the matrix A,
// in Ai/Ax [pfirst:plast-1], where plast = min (anz, pfirst+max_pchunk).
// Returns my_chunk_size = plast - pfirst, which is the size of the chunk
// operated on by this threadblock.

// The function GB_cuda_ek_slice behaves somewhat like GB_ek_slice.  The
// latter is for OpenMP parallelism on the CPU only, and it does not need to
// compute ks.

static __device__ __inline__ int64_t GB_cuda_ek_slice // returns my_chunk_size
(
    // inputs, not modified:
    const int64_t *Ap,          // array of size anvec+1
    const int64_t anvec,        // # of vectors in the matrix A
    const int64_t anz,          // # of entries in the sparse/hyper matrix A
    const int64_t pfirst,       // first entry in A to find k
    const int64_t max_pchunk,   // max # of entries in A to find k
    // output:
    int64_t *ks                 // k value for each pfirst:plast-1
)
{

    //--------------------------------------------------------------------------
    // determine the chunk for this threadblock
    //--------------------------------------------------------------------------

    // The slice for each threadblock contains entries pfirst:plast-1 of A.
    // The threadblock works on a chunk of entries in Ai/Ax [pfirst...plast-1].

    int64_t plast  = pfirst + max_pchunk ;
    plast = GB_IMIN (plast, anz) ;
    int64_t my_chunk_size = plast - pfirst ;

    //--------------------------------------------------------------------------
    // estimate the first and last vectors for this chunk
    //--------------------------------------------------------------------------

    // find kfirst, the first vector of the slice for this chunk.  kfirst is
    // the vector that owns the entry Ai [pfirst] and Ax [pfirst].  The search
    // does not need to be exact.

    int64_t kfirst = 0, kright = anvec ;
    GB_TRIM_BINARY_SEARCH (pfirst, Ap, kfirst, kright) ;

    // find klast, the last vector of the slice for this chunk.  klast is the
    // vector that owns the entry Ai [plast-1] and Ax [plast-1].  The search
    // does not have to be exact. 

    int64_t klast = kfirst ;
    kright = anvec ;
    GB_TRIM_BINARY_SEARCH (plast, Ap, klast, kright) ;

    // number of vectors in A for this chunk, where
    // Ap [kfirst:klast-1] will be searched.
    int64_t nk = klast - kfirst + 1 ;

    //--------------------------------------------------------------------------
    // search for k values for each entry pfirst:plast-1
    //--------------------------------------------------------------------------

    float slope = ((float) nk) / ((float) my_chunk_size) ;
    int64_t anvec1 = anvec - 1 ;
    for (int64_t kk = threadIdx.x ; kk < my_chunk_size ; kk += blockDim.x)
    {
        // get a rough estimate of k for the kkth entry in ks
        int64_t k = kfirst + (int64_t) (slope * ((float) kk)) ;
        // k cannot be smaller than kfirst, but might be bigger than
        // anvec-1, so ensure it is in the valid range, kfirst to anvec-1
        k = GB_IMIN (k, anvec1) ;
        // look for p in Ap, where p is in range pfirst:plast-1
        // where pfirst >= 0 and plast < anz
        int64_t p = kk + pfirst ;
        // linear-time search for the k value of the pth entry
        while (Ap [k+1] <= p) k++ ;
        while (Ap [k  ] >  p) k-- ;
        ks [kk] = k ;
    }

    //--------------------------------------------------------------------------
    // sync all threads and return result
    //--------------------------------------------------------------------------

    this_thread_block().sync() ;
    return (my_chunk_size) ;
}

