//------------------------------------------------------------------------------
// GB_serialize_to_blob: copy a set of blocks to the blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_serialize.h"

void GB_serialize_to_blob
(
    // input/output
    GB_void *blob,          // blocks are appended to the blob
    size_t *s_handle,       // location to append into the blob
    // input:
    GB_blocks *Blocks,      // Blocks: array of size nblocks
    int32_t nblocks,        // # of blocks
    int nthreads_max        // # of threads to use
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (blob != NULL) ;
    ASSERT (s_handle != NULL) ;
    ASSERT (nblocks >= 0) ;
    ASSERT ((nblocks > 0) == (Blocks != NULL)) ;
    ASSERT (nthreads_max > 0) ;

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    if (nblocks == 0)
    {
        // no blocks for this array
        return ;
    }

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    int nthreads = GB_IMIN (nthreads_max, nblocks) ;

    //--------------------------------------------------------------------------
    // get the current position for these Blocks in the blob
    //--------------------------------------------------------------------------

    size_t s = (*s_handle) ;

    //--------------------------------------------------------------------------
    // copy the Blocks array into the blob and the # of blocks
    //--------------------------------------------------------------------------

    // copy the # of blocks into the blob: a single int32_t scalar
    memcpy (blob + s, &nblocks, sizeof (int32_t)) ; s += sizeof (int32_t) ;

    // followed by three arrays
    int64_t *Ublock = blob + s ; s += sizeof (int64_t) * nblocks ;
    int64_t *Sblock = blob + s ; s += sizeof (int64_t) * nblocks ;
    int64_t *Method = blob + s ; s += sizeof (int32_t) * nblocks ;

    //--------------------------------------------------------------------------
    // copy the blocks into the blob
    //--------------------------------------------------------------------------

    int blockid ;
//  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
//  printf ("nblocks %d\n", nblocks) ;
    for (blockid = 0 ; blockid < nblocks ; blockid++)
    {
        // copy the scalar info into the 3 arrays:
        Ublock [blockid] = (int64_t) Blocks [blockid+1].uncompressed ;
        Sblock [blockid] = (int64_t) Blocks [blockid+1].compressed ;
        Method [blockid] = (int32_t) Blocks [blockid].method ;
        // copy the compressed block itself, of size s_size
        size_t s_start = Blocks [blockid].compressed ;
        size_t s_end   = Blocks [blockid+1].compressed ;
        size_t s_size  = s_end - s_start ;
        memcpy (blob + s + s_start, Blocks [blockid].p, s_size) ;
        // printf ("Ublock [%d] = %ld\n", blockid, Ublock [blockid]) ;
    }

    s += Blocks [nblocks].compressed ;

    //--------------------------------------------------------------------------
    // return the updated index into the blob
    //--------------------------------------------------------------------------

    (*s_handle) = s ;
}

