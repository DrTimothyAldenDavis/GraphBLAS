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
    // get the content of the matrix and fill the header
    //--------------------------------------------------------------------------

    // FIXME: do not use a struct for the header.  Just write individual
    // integers and floats and such, to avoid compiler-dependent packing.
    GB_blob_header header ;

    header.version = GxB_IMPLEMENTATION ;
    int64_t vlen = A->vlen ;
    int64_t vdim = A->vdim ;
    header.vlen = vlen ;
    header.vdim = vdim ;
    int64_t nvec = A->nvec ;
    header.nvec = nvec ;
    if (A->nvec_nonempty < 0) A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    header.nvec_nonempty = A->nvec_nonempty ;
    header.nvals = A->nvals ;

    // control settings
    header.hyper_switch = A->hyper_switch ;
    header.bitmap_switch = A->bitmap_switch ;
    header.sparsity_control = A->sparsity_control ;

    // current sparsity format
    int32_t sparsity = GB_sparsity (A) ;
    header.sparsity = sparsity ;

    header.is_csc = A->is_csc ;
    header.iso = A->iso ;
    bool iso = A->iso ;
    header.unused1 = 0 ;
    header.unused2 = 0 ;

    // the matrix has no pending work
    ASSERT (A->Pending == NULL) ;
    ASSERT (A->nzombies == 0) ;
    ASSERT (!A->jumbled) ;

    GrB_Type atype = A->type ;
    size_t typesize = atype->size ;
    GB_Type_code typecode = atype->code ;
    header.typecode = typecode ;
    header.typesize = typesize ;
    int64_t anz = GB_nnz (A) ;
    int64_t anz_held = GB_nnz_held (A) ;

    // determine the uncompressed sizes of Ap, Ah, Ab, Ai, and Ax
    size_t Ap_usage = 0 ;
    size_t Ah_usage = 0 ;
    size_t Ab_usage = 0 ;
    size_t Ai_usage = 0 ;
    size_t Ax_usage = 0 ;
    switch (sparsity)
    {
        case GxB_HYPERSPARSE : 
            Ah_usage = sizeof (GrB_Index) * nvec ;
        case GxB_SPARSE :
            Ap_usage = sizeof (GrB_Index) * (nvec+1) ;
            Ai_usage = sizeof (GrB_Index) * anz ;
            Ax_usage = typesize * (iso ? 1 : anz) ;
            break ;
        case GxB_BITMAP : 
            Ab_usage = sizeof (int8_t) * anz_held ;
        case GxB_FULL : 
            Ax_usage = typesize * (iso ? 1 : anz_held) ;
            break ;
        default: ;
    }

    //--------------------------------------------------------------------------
    // compress each array (Ap, Ah, Ab, Ai, and Ax)
    //--------------------------------------------------------------------------

    int32_t Ap_method, Ah_method, Ai_method, Ab_method, Ax_method ;

    // uint64
    GB_OK (GB_serialize_array (&Ap_blocks, &Ap_blocks_size, &Ap_nblocks,
        &Ap_method, (GB_void *) A->p, Ap_usage, method, Context)) ;
    GB_OK (GB_serialize_array (&Ah_blocks, &Ah_blocks_size, &Ah_nblocks,
        &Ah_method, (GB_void *) A->h, Ah_usage, method, Context)) ;
    GB_OK (GB_serialize_array (&Ai_blocks, &Ai_blocks_size, &Ai_nblocks,
        &Ai_method, (GB_void *) A->i, Ai_usage, method, Context)) ;

    // uint8
    GB_OK (GB_serialize_array (&Ab_blocks, &Ab_blocks_size, &Ab_nblocks,
        &Ab_method, (GB_void *) A->b, Ab_usage, method, Context)) ;

    // size depends on the matrix type
    GB_OK (GB_serialize_array (&Ax_blocks, &Ax_blocks_size, &Ax_nblocks,
        &Ax_method, (GB_void *) A->x, Ax_usage, method, Context)) ;

    //--------------------------------------------------------------------------
    // determine the size of the blob and allocate it
    //--------------------------------------------------------------------------

    size_t s =
        // header information
        sizeof (GB_blob_header)
        // typename for user-defined types
        + ((typecode == GB_UDT_code) ? GB_LEN : 0) ;

    // size of a compressed array:
    #define BSIZE(Blocks,nblocks)                                       \
        sizeof (int32_t)                    /* nblocks */               \
        + sizeof (int32_t)                  /* method used */           \
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

    header.blob_size = blob_size ;

    //--------------------------------------------------------------------------
    // copy the header and typename into the blob
    //--------------------------------------------------------------------------

    s = sizeof (GB_blob_header) ;
    memcpy (blob, &header, s) ;

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

    GB_serialize_to_blob (blob, &s, Ap_blocks, Ap_nblocks, Ap_method,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ah_blocks, Ah_nblocks, Ah_method,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ai_blocks, Ai_nblocks, Ai_method,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ab_blocks, Ab_nblocks, Ab_method,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ax_blocks, Ax_nblocks, Ax_method,
        nthreads_max) ;

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

