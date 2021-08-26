//------------------------------------------------------------------------------
// GB_deserialize_from_blob: uncompress a set of blocks from the blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_serialize.h"
#include "GB_lz4.h"

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE (&X64, X64_size) ;          \
    GB_FREE (&X, *X_size_handle) ;      \
}

GrB_Info GB_deserialize_from_blob
(
    // output:
    GB_void **X_handle,         // uncompressed output array
    size_t *X_size_handle,      // size of X
    // input:
    const GB_void *blob,
    size_t blob_size,
    // input/output:
    size_t *s_handle,           // location to write into the blob
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (blob != NULL) ;
    ASSERT (s_handle != NULL) ;
    ASSERT (X_handle != NULL) ;
    ASSERT (X_size_handle != NULL) ;
    (*X_handle) = NULL ;
    (*X_size_handle) = 0 ;
    uint64_t *X64 = NULL ;
    size_t X64_size = 0 ;

    //--------------------------------------------------------------------------
    // get the current position for these blocks in the blob for this array X
    //--------------------------------------------------------------------------

    size_t s = (*s_handle) ;

    //--------------------------------------------------------------------------
    // get the # of blocks for this array, and their sizes
    //--------------------------------------------------------------------------

    // copy the # of blocks from the blob: a single int32_t scalar
    int32_t nblocks ;
    memcpy (&nblocks, blob + s, sizeof (int32_t)) ; s += sizeof (int32_t) ;
    if (nblocks < 0)
    {
        // blob is invalid
        printf ("yeeeks! %d\n", __LINE__) ;
        return (GrB_PANIC) ;
    }

    // copy the compression method used from blob: a single int32_t scalar
    int32_t method_used ;
    memcpy (&method_used, blob + s, sizeof (int32_t)) ; s += sizeof (int32_t) ;

    // followed by two arrays: Ublock and Sblock
    int64_t *Ublock = blob + s ; s += sizeof (int64_t) * nblocks ;
    int64_t *Sblock = blob + s ; s += sizeof (int64_t) * nblocks ;

    if (s > blob_size)
    {
        // blob is invalid
        printf ("yeeeks! %d\n", __LINE__) ;
        return (GrB_PANIC) ;
    }

    //--------------------------------------------------------------------------
    // allocate the output array
    //--------------------------------------------------------------------------

    int64_t X_size = Ublock [nblocks-1] ;
    GB_void *X = GB_MALLOC (X_size, GB_void, X_size_handle) ;
    if (X == NULL)
    {
        // out of memory
        printf ("yeeeks! %d\n", __LINE__) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_IMIN (nthreads_max, nblocks) ;

    //--------------------------------------------------------------------------
    // decompress the blocks from the blob
    //--------------------------------------------------------------------------

    int blockid ;
    bool ok = true ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic) \
        reduction(&&:ok)
    for (blockid = 0 ; blockid < nblocks ; blockid++)
    {
        // get the scalar info from the 3 arrays:
        int64_t u_start = (blockid == 0) ? 0 : Ublock [blockid-1] ;
        int64_t u_end   = Ublock [blockid] ;
        int64_t s_start = (blockid == 0) ? 0 : Sblock [blockid-1] ;
        int64_t s_end   = Sblock [blockid] ;
        size_t s_size   = s_end - s_start ;

        // ensure s_start, s_end, u_start, and u_end are all valid,
        // to avoid accessing arrays out of bounds, if input is corrupted.
        if (u_start < 0 || u_end < 0 || s_start < 0 || s_end < 0 ||
            u_start >= u_end || s_start >= s_end ||
            s + s_start > blob_size || s + s_end > blob_size ||
            u_start > X_size || u_end > X_size)
        { 
            // blob is invalid
            printf ("BAD Yeeks %d\n", __LINE__) ;
            ok = false ;
        }
        else
        {
            // uncompress the compressed block of size s_size
            // from blob [s + s_start:s_end-1] into X [u_start:u_end-1]
            const char *src = (const char *) (blob + s + s_start) ;

            char *dst = (char *) (X + u_start) ;
            int src_size = (int) s_size ;
            int dst_size = (int) (u_end - u_start) ;
            if (method_used == GxB_COMPRESSION_NONE)
            {
                // no compression
                if (src_size != dst_size)
                { 
                    printf ("BAD TOO Yeeks %d\n", __LINE__) ;
                    ok = false ;
                }
                else
                { 
                    memcpy (dst, src, src_size) ;
                }
            }
            else
            { 
                // LZ4 compression
                int u ;
                // dump_blob (src, src_size) ;
                u = LZ4_decompress_safe (src, dst, src_size, dst_size) ;
                if (u <= 0)
                {
                    ok = false ;
                    printf ("ARG Yeeks %d %d\n", __LINE__, u) ;
                }
            }
        }
    }

    if (!ok)
    {
        // decompression failure
        GB_FREE_ALL ;
        printf ("yeeeks! %d\n", __LINE__) ;
        return (GrB_PANIC) ;
    }

    s += Sblock [nblocks-1] ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*X_handle) = X ;
    (*s_handle) = s ;
    return (GrB_SUCCESS) ;
}

