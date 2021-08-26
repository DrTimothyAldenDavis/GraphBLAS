//------------------------------------------------------------------------------
// GB_serialize_blocksizes_to_blob: copy the block sizes to the blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_serialize.h"

void GB_serialize_blocksizes_to_blob
(
    // output
    int64_t **Ublock_handle,    // location of the Ublock array
    int64_t **Sblock_handle,    // location of the Sblock array
    // input/output
    GB_void *blob,          // blocks are appended to the blob
    size_t *s_handle,       // location to append into the blob
    // input:
    GB_blocks *Blocks,      // Blocks: array of size nblocks
    int32_t nblocks         // # of blocks
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Ublock_handle != NULL) ;
    ASSERT (Sblock_handle != NULL) ;
    ASSERT (blob != NULL) ;
    ASSERT (s_handle != NULL) ;
    ASSERT (nblocks >= 0) ;
    ASSERT ((nblocks > 0) == (Blocks != NULL)) ;

    //--------------------------------------------------------------------------
    // copy the blocksizes into the blob
    //--------------------------------------------------------------------------

    size_t s = (*s_handle) ;
    int64_t *Ublock = blob + s ; s += sizeof (int64_t) * nblocks ;
    int64_t *Sblock = blob + s ; s += sizeof (int64_t) * nblocks ;

    for (int blockid = 0 ; blockid < nblocks ; blockid++)
    {
        // copy the scalar info into the 3 arrays:
        Ublock [blockid] = (int64_t) Blocks [blockid+1].uncompressed ;
        Sblock [blockid] = (int64_t) Blocks [blockid+1].compressed ;
    }

    //--------------------------------------------------------------------------
    // return Ublock, Sblock, and the updated index into the blob
    //--------------------------------------------------------------------------

    (*Ublock_handle) = Ublock ;
    (*Sblock_handle) = Sblock ;
    (*s_handle) = s ;
}

