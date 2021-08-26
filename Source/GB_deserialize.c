//------------------------------------------------------------------------------
// GB_deserialize: decompress and deserialize a blob into a GrB_Matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A parallel decompression of a serialized blob into a GrB_Matrix.

#include "GB.h"
#include "GB_serialize.h"

#define GB_FREE_ALL                         \
{                                           \
    GB_Matrix_free (&C) ;                   \
}

GrB_Info GB_deserialize             // deserialize a matrix from a blob
(
    // output:
    GrB_Matrix *Chandle,            // output matrix created from the blob
    // input:
    const GB_void *blob,            // serialized matrix 
    size_t blob_size,               // size of the blob
    GrB_Type user_type,             // type of matrix, if user-defined
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (blob != NULL && Chandle != NULL && blob_size > 0) ;
    (*Chandle) = NULL ;
    GrB_Matrix C = NULL ;

    //--------------------------------------------------------------------------
    // read the content of the header (160 bytes)
    //--------------------------------------------------------------------------

    #define MEMREAD(x,type) \
        type x ; memcpy (&x, blob + s, sizeof (type)) ; s += sizeof (type) ;

    size_t s = 0 ;

    // 7x8 = 56 bytes
    MEMREAD (blob_size2, size_t) ;
    MEMREAD (vlen, int64_t) ;
    MEMREAD (vdim, int64_t) ;
    MEMREAD (nvec, int64_t) ;
    MEMREAD (nvec_nonempty, int64_t) ;
    MEMREAD (nvals, int64_t) ;
    MEMREAD (typesize, int64_t) ;

    // 5x8 = 40 bytes
    MEMREAD (Cp_len, int64_t) ;
    MEMREAD (Ch_len, int64_t) ;
    MEMREAD (Cb_len, int64_t) ;
    MEMREAD (Ci_len, int64_t) ;
    MEMREAD (Cx_len, int64_t) ;

    // 6x4 = 24 bytes
    MEMREAD (version, int32_t) ;          MEMREAD (typecode, int32_t) ;
    MEMREAD (hyper_switch, float) ;       MEMREAD (bitmap_switch, float) ;
    MEMREAD (sparsity_control, int32_t) ; MEMREAD (sparsity_iso_csc, int32_t) ;

    // 10x4 = 40 bytes
    MEMREAD (Cp_nblocks, int32_t) ; MEMREAD (Cp_method, int32_t) ;
    MEMREAD (Ch_nblocks, int32_t) ; MEMREAD (Ch_method, int32_t) ;
    MEMREAD (Cb_nblocks, int32_t) ; MEMREAD (Cb_method, int32_t) ;
    MEMREAD (Ci_nblocks, int32_t) ; MEMREAD (Ci_method, int32_t) ;
    MEMREAD (Cx_nblocks, int32_t) ; MEMREAD (Cx_method, int32_t) ;

    int32_t sparsity = sparsity_iso_csc / 4 ;
    bool iso = ((sparsity_iso_csc & 2) == 1) ;
    bool is_csc = ((sparsity_iso_csc & 1) == 1) ;

    if (blob_size != blob_size2 || blob_size < s)
    {
        // blob is invalid
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // determine the matrix type
    //--------------------------------------------------------------------------

    GB_Type_code ccode = (GB_Type_code) typecode ;
    GrB_Type ctype = GB_code_type (ccode, user_type) ;

    // ensure the type has the right size
    if (ctype == NULL || ctype->size != typesize)
    {
        // blob is invalid
        return (GrB_INVALID_VALUE) ;
    }

    // 128 bytes, if present
    if (ccode == GB_UDT_code)
    {
        // ensure the user-defined type has the right name
        if ((blob_size < s + GB_LEN) ||
            (strncmp (blob + s, ctype->name, GB_LEN) != 0))
        {
            // blob is invalid
            return (GrB_INVALID_VALUE) ;
        }
        s += GB_LEN ;
    }

    //--------------------------------------------------------------------------
    // get the compressed block sizes from the blob for each array
    //--------------------------------------------------------------------------

    #define MEMREADS(S,n) \
        int64_t *S = (int64_t *) (blob + s) ; s += n * sizeof (int64_t) ;
    MEMREADS (Cp_Sblocks, Cp_nblocks) ;
    MEMREADS (Ch_Sblocks, Ch_nblocks) ;
    MEMREADS (Cb_Sblocks, Cb_nblocks) ;
    MEMREADS (Ci_Sblocks, Ci_nblocks) ;
    MEMREADS (Cx_Sblocks, Cx_nblocks) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    // allocate the matrix with info from the header
    GB_OK (GB_new (Chandle, false, ctype, vlen, vdim, GB_Ap_null, is_csc,
        sparsity, hyper_switch, nvec, Context)) ;

    C = (*Chandle) ;
    C->nvec = nvec ;
    C->nvec_nonempty = nvec_nonempty ;
    C->nvals = nvals ;
    C->bitmap_switch = bitmap_switch ;
    C->sparsity_control = sparsity_control ;
    C->iso = iso ;

    // the matrix has no pending work
    ASSERT (C->Pending == NULL) ;
    ASSERT (C->nzombies == 0) ;
    ASSERT (!C->jumbled) ;

    //--------------------------------------------------------------------------
    // decompress each array (Cp, Ch, Cb, Ci, and Cx)
    //--------------------------------------------------------------------------

    switch (sparsity)
    {
        case GxB_HYPERSPARSE : 

            // decompress Cp, Ch, and Ci
            GB_OK (GB_deserialize_from_blob (&(C->p), &(C->p_size), Cp_len,
                blob, blob_size, Cp_Sblocks, Cp_nblocks, Cp_method,
                &s, Context)) ;
            GB_OK (GB_deserialize_from_blob (&(C->h), &(C->h_size), Ch_len,
                blob, blob_size, Ch_Sblocks, Ch_nblocks, Ch_method,
                &s, Context)) ;
            GB_OK (GB_deserialize_from_blob (&(C->i), &(C->i_size), Ci_len,
                blob, blob_size, Ci_Sblocks, Ci_nblocks, Ci_method,
                &s, Context)) ;
            break ;

        case GxB_SPARSE :

            // decompress Cp and Ci
            GB_OK (GB_deserialize_from_blob (&(C->p), &(C->p_size), Cp_len,
                blob, blob_size, Cp_Sblocks, Cp_nblocks, Cp_method,
                &s, Context)) ;
            GB_OK (GB_deserialize_from_blob (&(C->i), &(C->i_size), Ci_len,
                blob, blob_size, Ci_Sblocks, Ci_nblocks, Ci_method,
                &s, Context)) ;
            break ;

        case GxB_BITMAP : 

            // decompress Cb
            GB_OK (GB_deserialize_from_blob (&(C->b), &(C->b_size), Cb_len,
                blob, blob_size, Cb_Sblocks, Cb_nblocks, Cb_method,
                &s, Context)) ;
            break ;

        case GxB_FULL : 
            break ;
        default: ;
    }

    // decompress Cx
    GB_OK (GB_deserialize_from_blob (&(C->x), &(C->x_size), Cx_len,
        blob, blob_size, Cx_Sblocks, Cx_nblocks, Cx_method, &s, Context)) ;

    //--------------------------------------------------------------------------
    // finalize matrix return result
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C from deserialize", GB0) ;
    return (GrB_SUCCESS) ;
}

