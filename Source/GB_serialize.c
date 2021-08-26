//------------------------------------------------------------------------------
// GB_serialize: compress and serialize a GrB_Matrix into a blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A parallel compression method for a GrB_Matrix.  The input matrix may have
// shallow components; the output is unaffected.  The output blob is allocated
// on output, with size given by blob_size.

#include "GB.h"
#include "GB_serialize.h"

#define GB_FREE_WORK                            \
{                                               \
    GB_FREE (&Ap_Sblocks, Ap_Sblocks_size) ;    \
    GB_FREE (&Ah_Sblocks, Ah_Sblocks_size) ;    \
    GB_FREE (&Ab_Sblocks, Ab_Sblocks_size) ;    \
    GB_FREE (&Ai_Sblocks, Ai_Sblocks_size) ;    \
    GB_FREE (&Ax_Sblocks, Ax_Sblocks_size) ;    \
    GB_serialize_free_blocks (&Ap_Blocks, Ap_Blocks_size, Ap_nblocks, Context);\
    GB_serialize_free_blocks (&Ah_Blocks, Ah_Blocks_size, Ah_nblocks, Context);\
    GB_serialize_free_blocks (&Ab_Blocks, Ab_Blocks_size, Ab_nblocks, Context);\
    GB_serialize_free_blocks (&Ai_Blocks, Ai_Blocks_size, Ai_nblocks, Context);\
    GB_serialize_free_blocks (&Ax_Blocks, Ax_Blocks_size, Ax_nblocks, Context);\
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORK ;                          \
    GB_FREE (&blob, blob_size) ;            \
}

GrB_Info GB_serialize               // serialize a matrix into a blob
(
    // output:
    GB_void **blob_handle,          // serialized matrix, allocated on output
    size_t *blob_size_handle,       // size of the blob
    // input:
    const GrB_Matrix A,             // matrix to serialize
    int32_t method,                 // method to use
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (blob_handle != NULL && blob_size_handle != NULL) ;
    ASSERT_MATRIX_OK (A, "A for serialize", GB0) ;
    (*blob_handle) = NULL ;
    (*blob_size_handle) = 0 ;

    GB_void *blob = NULL ; size_t blob_size = 0 ;
    GB_blocks *Ap_Blocks = NULL ; size_t Ap_Blocks_size = 0 ;
    GB_blocks *Ah_Blocks = NULL ; size_t Ah_Blocks_size = 0 ;
    GB_blocks *Ab_Blocks = NULL ; size_t Ab_Blocks_size = 0 ;
    GB_blocks *Ai_Blocks = NULL ; size_t Ai_Blocks_size = 0 ;
    GB_blocks *Ax_Blocks = NULL ; size_t Ax_Blocks_size = 0 ;
    int64_t *Ap_Sblocks = NULL ; size_t Ap_Sblocks_size = 0 ;
    int64_t *Ah_Sblocks = NULL ; size_t Ah_Sblocks_size = 0 ;
    int64_t *Ab_Sblocks = NULL ; size_t Ab_Sblocks_size = 0 ;
    int64_t *Ai_Sblocks = NULL ; size_t Ai_Sblocks_size = 0 ;
    int64_t *Ax_Sblocks = NULL ; size_t Ax_Sblocks_size = 0 ;
    int32_t Ap_nblocks = 0 ;
    int32_t Ah_nblocks = 0 ;
    int32_t Ab_nblocks = 0 ;
    int32_t Ai_nblocks = 0 ;
    int32_t Ax_nblocks = 0 ;

    // method = GxB_COMPRESSION_NONE ;

    //--------------------------------------------------------------------------
    // ensure all pending work is finished
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (A) ;

    //--------------------------------------------------------------------------
    // determine maximum # of threads
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // get the content of the matrix
    //--------------------------------------------------------------------------

    int32_t version = GxB_IMPLEMENTATION ;
    int64_t vlen = A->vlen ;
    int64_t vdim = A->vdim ;
    int64_t nvec = A->nvec ;
    int64_t nvals = A->nvals ;
    // FIXME: nvec_nonempty not threadsafe
    if (A->nvec_nonempty < 0) A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    int64_t nvec_nonempty = A->nvec_nonempty ;
    int32_t sparsity = GB_sparsity (A) ;
    bool iso = A->iso ;
    float hyper_switch = A->hyper_switch ;
    float bitmap_switch = A->bitmap_switch ;
    int32_t sparsity_control = A->sparsity_control ;
    // the matrix has no pending work
    ASSERT (A->Pending == NULL) ;
    ASSERT (A->nzombies == 0) ;
    ASSERT (!A->jumbled) ;
    GrB_Type atype = A->type ;
    size_t typesize = atype->size ;
    int32_t typecode = (int32_t) (atype->code) ;
    int64_t anz = GB_nnz (A) ;
    int64_t anz_held = GB_nnz_held (A) ;

    // determine the uncompressed sizes of Ap, Ah, Ab, Ai, and Ax
    int64_t Ap_len = 0 ;
    int64_t Ah_len = 0 ;
    int64_t Ab_len = 0 ;
    int64_t Ai_len = 0 ;
    int64_t Ax_len = 0 ;
    switch (sparsity)
    {
        case GxB_HYPERSPARSE : 
            Ah_len = sizeof (GrB_Index) * nvec ;
        case GxB_SPARSE :
            Ap_len = sizeof (GrB_Index) * (nvec+1) ;
            Ai_len = sizeof (GrB_Index) * anz ;
            Ax_len = typesize * (iso ? 1 : anz) ;
            break ;
        case GxB_BITMAP : 
            Ab_len = sizeof (int8_t) * anz_held ;
        case GxB_FULL : 
            Ax_len = typesize * (iso ? 1 : anz_held) ;
            break ;
        default: ;
    }

    //--------------------------------------------------------------------------
    // compress each array (Ap, Ah, Ab, Ai, and Ax)
    //--------------------------------------------------------------------------

    int32_t Ap_method, Ah_method, Ab_method, Ai_method, Ax_method ;
    GB_OK (GB_serialize_array (&Ap_Blocks, &Ap_Blocks_size,
        &Ap_Sblocks, &Ap_Sblocks_size, &Ap_nblocks, &Ap_method,
        (GB_void *) A->p, Ap_len, method, Context)) ;
    GB_OK (GB_serialize_array (&Ah_Blocks, &Ah_Blocks_size,
        &Ah_Sblocks, &Ah_Sblocks_size, &Ah_nblocks, &Ah_method,
        (GB_void *) A->h, Ah_len, method, Context)) ;
    GB_OK (GB_serialize_array (&Ab_Blocks, &Ab_Blocks_size,
        &Ab_Sblocks, &Ab_Sblocks_size, &Ab_nblocks, &Ab_method,
        (GB_void *) A->b, Ab_len, method, Context)) ;
    GB_OK (GB_serialize_array (&Ai_Blocks, &Ai_Blocks_size,
        &Ai_Sblocks, &Ai_Sblocks_size, &Ai_nblocks, &Ai_method,
        (GB_void *) A->i, Ai_len, method, Context)) ;
    GB_OK (GB_serialize_array (&Ax_Blocks, &Ax_Blocks_size,
        &Ax_Sblocks, &Ax_Sblocks_size, &Ax_nblocks, &Ax_method,
        (GB_void *) A->x, Ax_len, method, Context)) ;

    //--------------------------------------------------------------------------
    // determine the size of the blob and allocate it
    //--------------------------------------------------------------------------

    size_t s =
        // header information
        sizeof (size_t)             // blob_size
        + 11 * sizeof (int64_t)     // vlen, vdim, nvec, nvec_nonempty, nvals,
                                    // typesize, A[phbix]_len
        + 4 * sizeof (int32_t)      // version, typecode, sparsity_control,
                                    // sparsity_iso_csc
        + 2 * sizeof (float)        // hyper_switch, bitmap_switch
        + 10 * sizeof (int32_t)     // A[phbix]_nblocks, A[phbix]_method
        // Sblocks for each array
        + Ap_nblocks * sizeof (int64_t)     // Ap_Sblocks [1:Ap_nblocks]
        + Ah_nblocks * sizeof (int64_t)     // Ah_Sblocks [1:Ah_nblocks]
        + Ab_nblocks * sizeof (int64_t)     // Ab_Sblocks [1:Ab_nblocks]
        + Ai_nblocks * sizeof (int64_t)     // Ai_Sblocks [1:Ai_nblocks]
        + Ax_nblocks * sizeof (int64_t)     // Ax_Sblocks [1:Ax_nblocks]
        // typename for user-defined types
        + ((typecode == GB_UDT_code) ? GB_LEN : 0) ;

    // size of compressed arrays Ap, Ah, Ab, Ai, and Ax in the blob
    if (Ap_nblocks > 0) s += Ap_Sblocks [Ap_nblocks] ;
    if (Ah_nblocks > 0) s += Ah_Sblocks [Ah_nblocks] ;
    if (Ab_nblocks > 0) s += Ab_Sblocks [Ab_nblocks] ;
    if (Ai_nblocks > 0) s += Ai_Sblocks [Ai_nblocks] ;
    if (Ax_nblocks > 0) s += Ax_Sblocks [Ax_nblocks] ;

    // GB_MALLOC may decide to increase the blob from size s bytes to blob_size
    blob = GB_MALLOC (s, GB_void, &blob_size) ;
    if (blob == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // write the header and typename into the blob
    //--------------------------------------------------------------------------

    // 160 bytes, plus 128 bytes for user-defined types 

    #define MEMWRITE(x,type) \
        memcpy (blob + s, &(x), sizeof (type)) ; s += sizeof (type) ;

    s = 0 ;
    int32_t sparsity_iso_csc = (4 * sparsity) + (iso ? 2 : 0) +
        (A->is_csc ? 1 : 0) ;

    // 7x8 = 56 bytes   (1 size_t, 6 int64)
    MEMWRITE (blob_size, size_t) ;
    MEMWRITE (vlen, int64_t) ;
    MEMWRITE (vdim, int64_t) ;
    MEMWRITE (nvec, int64_t) ;
    MEMWRITE (nvec_nonempty, int64_t) ;
    MEMWRITE (nvals, int64_t) ;
    MEMWRITE (typesize, int64_t) ;

    // 5x8 = 40 bytes (5 int64)
    MEMWRITE (Ap_len, int64_t) ;
    MEMWRITE (Ah_len, int64_t) ;
    MEMWRITE (Ab_len, int64_t) ;
    MEMWRITE (Ai_len, int64_t) ;
    MEMWRITE (Ax_len, int64_t) ;

    // 6x4 = 24 bytes
    MEMWRITE (version, int32_t) ;          MEMWRITE (typecode, int32_t) ;
    MEMWRITE (hyper_switch, float) ;       MEMWRITE (bitmap_switch, float) ;
    MEMWRITE (sparsity_control, int32_t) ; MEMWRITE (sparsity_iso_csc, int32_t);

    // 10x4 = 40 bytes
    MEMWRITE (Ap_nblocks, int32_t) ; MEMWRITE (Ap_method, int32_t) ;
    MEMWRITE (Ah_nblocks, int32_t) ; MEMWRITE (Ah_method, int32_t) ;
    MEMWRITE (Ab_nblocks, int32_t) ; MEMWRITE (Ab_method, int32_t) ;
    MEMWRITE (Ai_nblocks, int32_t) ; MEMWRITE (Ai_method, int32_t) ;
    MEMWRITE (Ax_nblocks, int32_t) ; MEMWRITE (Ax_method, int32_t) ;

    // 128 bytes, if present
    if (typecode == GB_UDT_code)
    {
        // only copy the typename for user-defined types
        memset (blob + s, 0, GB_LEN) ;
        strncpy (blob + s, atype->name, GB_LEN-1) ;
        s += GB_LEN ;
    }

    //--------------------------------------------------------------------------
    // copy the compressed arrays into the blob
    //--------------------------------------------------------------------------

    #define MEMWRITES(S,n) \
        if (n > 0)                                              \
        {                                                       \
            memcpy (blob + s, S + 1, n * sizeof (int64_t)) ;    \
            s += n * sizeof (int64_t) ;                         \
        }

    // 8 * (# blocks for Ap, Ah, Ab, Ai, Ax)
    MEMWRITES (Ap_Sblocks, Ap_nblocks) ;
    MEMWRITES (Ah_Sblocks, Ah_nblocks) ;
    MEMWRITES (Ab_Sblocks, Ab_nblocks) ;
    MEMWRITES (Ai_Sblocks, Ai_nblocks) ;
    MEMWRITES (Ax_Sblocks, Ax_nblocks) ;

    // FIXME: round up each of the 5 arrays to a multiple of 64 bytes,
    // or each internal compressed block.

    GB_serialize_to_blob (blob, &s, Ap_Blocks, Ap_Sblocks+1, Ap_nblocks,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ah_Blocks, Ah_Sblocks+1, Ah_nblocks,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ab_Blocks, Ab_Sblocks+1, Ab_nblocks,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ai_Blocks, Ai_Sblocks+1, Ai_nblocks,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ax_Blocks, Ax_Sblocks+1, Ax_nblocks,
        nthreads_max) ;

    // the blob is at least of size s, but might be slightly larger, as
    // determined by GB_MALLOC, so zero out any unused bytes
    ASSERT (s <= blob_size) ;
    if (s < blob_size) memset (blob + s, 0, blob_size - s) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    // giving the blob to the user; remove it from the list of malloc'd blocks
    #ifdef GB_MEMDUMP
    printf ("removing blob %p size %ld from memtable\n", blob, blob_size) ;
    #endif
    GB_Global_memtable_remove (blob) ;
    GB_FREE_WORK ;
    (*blob_handle) = blob ;
    (*blob_size_handle) = blob_size ;
    return (GrB_SUCCESS) ;
}

