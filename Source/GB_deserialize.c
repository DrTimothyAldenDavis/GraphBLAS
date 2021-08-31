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
    GB_Matrix_free (&T) ;                   \
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
    ASSERT (blob != NULL && Chandle != NULL) ;
    (*Chandle) = NULL ;
    GrB_Matrix C = NULL, T = NULL ;

    //--------------------------------------------------------------------------
    // read the content of the header (160 bytes)
    //--------------------------------------------------------------------------

    size_t s = 0 ;

    if (blob_size < GB_BLOB_HEADER_SIZE)
    { 
        // blob is invalid
        return (GrB_INVALID_VALUE) ;
    }

    GB_BLOB_READ (blob_size2, size_t) ;
    GB_BLOB_READ (typecode, int32_t) ;

    if (blob_size != blob_size2
        || typecode < GB_BOOL_code || typecode > GB_UDT_code
        || (typecode == GB_UDT_code &&
            blob_size < GB_BLOB_HEADER_SIZE + GxB_MAX_NAME_LEN))
    { 
        // blob is invalid
        return (GrB_INVALID_VALUE) ;
    }

    GB_BLOB_READ (version, int32_t) ;
    GB_BLOB_READ (vlen, int64_t) ;
    GB_BLOB_READ (vdim, int64_t) ;
    GB_BLOB_READ (nvec, int64_t) ;
    GB_BLOB_READ (nvec_nonempty, int64_t) ;     ASSERT (nvec_nonempty >= 0) ;
    GB_BLOB_READ (nvals, int64_t) ;
    GB_BLOB_READ (typesize, int64_t) ;
    GB_BLOB_READ (Cp_len, int64_t) ;
    GB_BLOB_READ (Ch_len, int64_t) ;
    GB_BLOB_READ (Cb_len, int64_t) ;
    GB_BLOB_READ (Ci_len, int64_t) ;
    GB_BLOB_READ (Cx_len, int64_t) ;
    GB_BLOB_READ (hyper_switch, float) ;
    GB_BLOB_READ (bitmap_switch, float) ;
    GB_BLOB_READ (sparsity_control, int32_t) ;
    GB_BLOB_READ (sparsity_iso_csc, int32_t) ;
    GB_BLOB_READ (Cp_nblocks, int32_t) ; GB_BLOB_READ (Cp_method, int32_t) ;
    GB_BLOB_READ (Ch_nblocks, int32_t) ; GB_BLOB_READ (Ch_method, int32_t) ;
    GB_BLOB_READ (Cb_nblocks, int32_t) ; GB_BLOB_READ (Cb_method, int32_t) ;
    GB_BLOB_READ (Ci_nblocks, int32_t) ; GB_BLOB_READ (Ci_method, int32_t) ;
    GB_BLOB_READ (Cx_nblocks, int32_t) ; GB_BLOB_READ (Cx_method, int32_t) ;

    int32_t sparsity = sparsity_iso_csc / 4 ;
    bool iso = ((sparsity_iso_csc & 2) == 1) ;
    bool is_csc = ((sparsity_iso_csc & 1) == 1) ;

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
        ASSERT (ctype == user_type) ;
        if (strncmp (blob + s, ctype->name, GxB_MAX_NAME_LEN) != 0)
        { 
            // blob is invalid
            return (GrB_INVALID_VALUE) ;
        }
        s += GxB_MAX_NAME_LEN ;
    }

    //--------------------------------------------------------------------------
    // get the compressed block sizes from the blob for each array
    //--------------------------------------------------------------------------

    GB_BLOB_READS (Cp_Sblocks, Cp_nblocks) ;
    GB_BLOB_READS (Ch_Sblocks, Ch_nblocks) ;
    GB_BLOB_READS (Cb_Sblocks, Cb_nblocks) ;
    GB_BLOB_READS (Ci_Sblocks, Ci_nblocks) ;
    GB_BLOB_READS (Cx_Sblocks, Cx_nblocks) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    // allocate the matrix with info from the header
    GB_OK (GB_new (&C, false, ctype, vlen, vdim, GB_Ap_null, is_csc,
        sparsity, hyper_switch, nvec, Context)) ;

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
    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C from deserialize (before typecast)", GB0) ;

    //--------------------------------------------------------------------------
    // typecast if requested user_type differs from ctype
    //--------------------------------------------------------------------------

    if (ctype != user_type && user_type != NULL)
    { 
        printf ("cast the blob from %s to %s:\n",
            C->type->name, user_type->name) ;
        // T = empty matrix with the final user_type.  All content except the
        // header of T itself will be overwritten by C.
        ASSERT (ccode != GB_UDT_code) ;
        GB_OK (GB_new (&T, false, user_type, 0, 0, GB_Ap_null, false,
            GxB_AUTO_SPARSITY, GB_HYPER_SWITCH_DEFAULT, 0, Context)) ;
        // transplant C into T, and typecast to user_type
        GB_OK (GB_transplant (T, user_type, &C, Context)) ;
        ASSERT (C == NULL) ;
        (*Chandle) = T ;
    }
    else
    { 
        // return result as-is
        (*Chandle) = C ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (*Chandle, "Final result from deserialize", GB0) ;
    return (GrB_SUCCESS) ;
}

