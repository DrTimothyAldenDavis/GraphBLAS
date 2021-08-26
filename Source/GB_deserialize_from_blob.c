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
    GB_FREE (&X, *X_size_handle) ;      \
}

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

    //--------------------------------------------------------------------------
    // allocate the output array
    //--------------------------------------------------------------------------

    int64_t X_size = Ublock [nblocks-1] ;
    ASSERT (X_size == X_len) ;
    GB_void *X = GB_MALLOC (X_len, GB_void, X_size_handle) ;
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

    size_t s = (*s_handle) ;
    int blockid ;
    bool ok = true ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic) \
        reduction(&&:ok)
    for (blockid = 0 ; blockid < nblocks ; blockid++)
    {
        // get the scalar info from the 3 arrays:
        int64_t u_start = (blockid == 0) ? 0 : Ublock [blockid-1] ;
        int64_t u_end   = Ublock [blockid] ;

        int64_t kstart, kend ;
        GB_PARTITION (kstart, kend, X_len, blockid, nblocks) ;
        ASSERT (kstart == u_start) ;
        ASSERT (kend == u_end) ;

        int64_t s_start = (blockid == 0) ? 0 : Sblock [blockid-1] ;
        int64_t s_end   = Sblock [blockid] ;
        size_t s_size   = s_end - s_start ;

        // ensure s_start, s_end, u_start, and u_end are all valid,
        // to avoid accessing arrays out of bounds, if input is corrupted.
        if (u_start < 0 || u_end < 0 || s_start < 0 || s_end < 0 ||
            u_start >= u_end || s_start >= s_end ||
            s + s_start > blob_size || s + s_end > blob_size ||
            u_start > X_len || u_end > X_len)
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

