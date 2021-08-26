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

#define GB_FREE_WORK                        \
{                                           \
    GB_serialize_free_blocks (&Ap_blocks, Ap_blocks_size, Ap_nblocks, Context);\
    GB_serialize_free_blocks (&Ah_blocks, Ah_blocks_size, Ah_nblocks, Context);\
    GB_serialize_free_blocks (&Ab_blocks, Ab_blocks_size, Ab_nblocks, Context);\
    GB_serialize_free_blocks (&Ai_blocks, Ai_blocks_size, Ai_nblocks, Context);\
    GB_serialize_free_blocks (&Ax_blocks, Ax_blocks_size, Ax_nblocks, Context);\
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

    GB_void *blob = NULL ;
    size_t blob_size = 0 ;
    GB_blocks *Ap_blocks = NULL ; int32_t Ap_nblocks = 0 ;
    GB_blocks *Ah_blocks = NULL ; int32_t Ah_nblocks = 0 ;
    GB_blocks *Ab_blocks = NULL ; int32_t Ab_nblocks = 0 ;
    GB_blocks *Ai_blocks = NULL ; int32_t Ai_nblocks = 0 ;
    GB_blocks *Ax_blocks = NULL ; int32_t Ax_nblocks = 0 ;
    size_t Ap_blocks_size = 0 ;
    size_t Ah_blocks_size = 0 ;
    size_t Ab_blocks_size = 0 ;
    size_t Ai_blocks_size = 0 ;
    size_t Ax_blocks_size = 0 ;

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

    int32_t Ap_method, Ah_method, Ai_method, Ab_method, Ax_method ;

    // uint64
    GB_OK (GB_serialize_array (&Ap_blocks, &Ap_blocks_size, &Ap_nblocks,
        &Ap_method, (GB_void *) A->p, Ap_len, method, Context)) ;
    GB_OK (GB_serialize_array (&Ah_blocks, &Ah_blocks_size, &Ah_nblocks,
        &Ah_method, (GB_void *) A->h, Ah_len, method, Context)) ;
    GB_OK (GB_serialize_array (&Ai_blocks, &Ai_blocks_size, &Ai_nblocks,
        &Ai_method, (GB_void *) A->i, Ai_len, method, Context)) ;

    // uint8
    GB_OK (GB_serialize_array (&Ab_blocks, &Ab_blocks_size, &Ab_nblocks,
        &Ab_method, (GB_void *) A->b, Ab_len, method, Context)) ;

    // size depends on the matrix type
    GB_OK (GB_serialize_array (&Ax_blocks, &Ax_blocks_size, &Ax_nblocks,
        &Ax_method, (GB_void *) A->x, Ax_len, method, Context)) ;

    //--------------------------------------------------------------------------
    // determine the size of the blob and allocate it
    //--------------------------------------------------------------------------

    size_t s =
        // header information
        sizeof (size_t) + 11 * sizeof (int64_t)
        + 4 * sizeof (int32_t)
        + 2 * sizeof (float)
        + 10 * sizeof (int32_t)
        // typename for user-defined types
        + ((typecode == GB_UDT_code) ? GB_LEN : 0) ;

    // size of a compressed array:
    #define BSIZE(Blocks,nblocks)                                       \
        + (sizeof (int64_t)) * nblocks      /* Ublock array */          \
        + (sizeof (int64_t)) * nblocks      /* Sblock array */          \
        + Blocks [nblocks].compressed       /* compressed blocks */

    // size of Ap, Ah, Ab, Ai, and Ax in the blob
    if (Ap_nblocks > 0) s += BSIZE (Ap_blocks, Ap_nblocks) ;
    if (Ah_nblocks > 0) s += BSIZE (Ah_blocks, Ah_nblocks) ;
    if (Ab_nblocks > 0) s += BSIZE (Ab_blocks, Ab_nblocks) ;
    if (Ai_nblocks > 0) s += BSIZE (Ai_blocks, Ai_nblocks) ;
    if (Ax_nblocks > 0) s += BSIZE (Ax_blocks, Ax_nblocks) ;

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

    // TODO: use phbix order throughout

    // 5x8 = 40 bytes (5 int64)
    MEMWRITE (Ap_len, int64_t) ;
    MEMWRITE (Ah_len, int64_t) ;
    MEMWRITE (Ai_len, int64_t) ;
    MEMWRITE (Ab_len, int64_t) ;
    MEMWRITE (Ax_len, int64_t) ;

    // 6x4 = 24 bytes
    MEMWRITE (version, int32_t) ;          MEMWRITE (typecode, int32_t) ;
    MEMWRITE (hyper_switch, float) ;       MEMWRITE (bitmap_switch, float) ;
    MEMWRITE (sparsity_control, int32_t) ; MEMWRITE (sparsity_iso_csc, int32_t);

    // 10x4 = 40 bytes
    MEMWRITE (Ap_nblocks, int32_t) ; MEMWRITE (Ap_method, int32_t) ;
    MEMWRITE (Ah_nblocks, int32_t) ; MEMWRITE (Ah_method, int32_t) ;
    MEMWRITE (Ai_nblocks, int32_t) ; MEMWRITE (Ai_method, int32_t) ;
    MEMWRITE (Ab_nblocks, int32_t) ; MEMWRITE (Ab_method, int32_t) ;
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

    // FIXME: round up each of the 5 arrays to a multiple of 64 bytes,
    // or each internal block.

    int64_t *Ap_Ublock, *Ap_Sblock ;
    int64_t *Ah_Ublock, *Ah_Sblock ;
    int64_t *Ai_Ublock, *Ai_Sblock ;
    int64_t *Ab_Ublock, *Ab_Sblock ;
    int64_t *Ax_Ublock, *Ax_Sblock ;

    // 16 * (blocks for Ap, Ah, Ai, Ab, Ax)
    GB_serialize_blocksizes_to_blob (&Ap_Ublock, &Ap_Sblock, blob, &s, Ap_blocks, Ap_nblocks) ;
    GB_serialize_blocksizes_to_blob (&Ah_Ublock, &Ah_Sblock, blob, &s, Ah_blocks, Ah_nblocks) ;
    GB_serialize_blocksizes_to_blob (&Ai_Ublock, &Ai_Sblock, blob, &s, Ai_blocks, Ai_nblocks) ;
    GB_serialize_blocksizes_to_blob (&Ab_Ublock, &Ab_Sblock, blob, &s, Ab_blocks, Ab_nblocks) ;
    GB_serialize_blocksizes_to_blob (&Ax_Ublock, &Ax_Sblock, blob, &s, Ax_blocks, Ax_nblocks) ;

    printf (" s %lu  (%d %d %d %d %d)\n", s,
        Ap_nblocks, Ah_nblocks, Ai_nblocks, Ab_nblocks, Ax_nblocks) ;

    GB_serialize_to_blob (blob, &s, Ap_blocks, Ap_Sblock, Ap_nblocks, nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ah_blocks, Ah_Sblock, Ah_nblocks, nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ai_blocks, Ai_Sblock, Ai_nblocks, nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ab_blocks, Ab_Sblock, Ab_nblocks, nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ax_blocks, Ax_Sblock, Ax_nblocks, nthreads_max) ;

    // the blob is at least of size s, but might be slightly larger,
    // so zero out any unused bytes
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

