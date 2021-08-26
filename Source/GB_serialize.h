//------------------------------------------------------------------------------
// GB_serialize.h: definitions for GB_serialize_* and deserialize methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SERIALIZE_H
#define GB_SERIALIZE_H

GrB_Info GB_serialize               // serialize a matrix into a blob
(
    // output:
    GB_void **blob_handle,          // serialized matrix, allocated on output
    size_t *blob_size_handle,       // size of the blob
    // input:
    const GrB_Matrix A,             // matrix to serialize
    int32_t method,                 // method to use
    GB_Context Context
) ;

GrB_Info GB_deserialize             // deserialize a matrix from a blob
(
    // output:
    GrB_Matrix *Chandle,            // output matrix created from the blob
    // input:
    const GB_void *blob,            // serialized matrix 
    size_t blob_size,               // size of the blob
    GrB_Type user_type,             // type of matrix, if user-defined
    GB_Context Context
) ;

typedef struct
{
    void *p ;                       // pointer to the compressed block
    size_t p_size ;                 // size of compressed block, or zero
                                    // if p is not malloc'ed
    // after the blocks are compressed, these 2 terms are overwritten with
    // their cumulative sum:
    size_t uncompressed ;           // original size of the block
    size_t compressed ;             // size of the block when compressed
}
GB_blocks ;

GrB_Info GB_serialize_array
(
    // output:
    GB_blocks **Blocks_handle,          // Blocks: array of size nblocks+1
    size_t *Blocks_size_handle,         // size of Blocks
    int32_t *nblocks_handle,            // # of blocks
    int32_t *method_used,               // method used
    // input:
    GB_void *X,                         // input array of size len
    int64_t len,                        // size of X, in bytes
    int32_t method,                     // compression method requested
    GB_Context Context
) ;

void GB_serialize_free_blocks
(
    GB_blocks **Blocks_handle,      // array of size nblocks
    size_t Blocks_size,             // size of Blocks
    int32_t nblocks,                // # of blocks, or zero if no blocks
    GB_Context Context
) ;

void GB_serialize_to_blob
(
    // input/output
    GB_void *blob,          // blocks are appended to the blob
    size_t *s_handle,       // location to append into the blob
    // input:
    GB_blocks *Blocks,      // Blocks: array of size nblocks+1
    int64_t *Sblock,        // array of size nblocks
    int32_t nblocks,        // # of blocks
    int nthreads_max        // # of threads to use
) ;

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
) ;

GrB_Info GB_deserialize_from_blob
(
    // output:
    GB_void **X_handle,         // uncompressed output array
    size_t *X_size_handle,      // size of X as allocated
    // input:
    int64_t X_len,              // size of X in bytes
    const GB_void *blob,
    size_t blob_size,
    int64_t *Ublock,            // array of size nblocks
    int64_t *Sblock,            // array of size nblocks
    int32_t nblocks,
    int32_t method_used,
    // input/output:
    size_t *s_handle,           // location to write into the blob
    GB_Context Context
) ;

#endif

