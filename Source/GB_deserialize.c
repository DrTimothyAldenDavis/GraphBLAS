//------------------------------------------------------------------------------
// GB_deserialize: uncompress and deserialize a blob into a GrB_Matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A parallel compression method for a GrB_Matrix.  The input matrix may have
// shallow components; the output is unaffected.  The output blob is allocated
// on output, with size given by blob_size.

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
    GrB_Matrix C = NULL ;

    //--------------------------------------------------------------------------
    // get the content of the header and allocate the output matrix C
    //--------------------------------------------------------------------------

    GB_blob_header *header = (GB_blob_header *) blob ;
    size_t s = sizeof (GB_blob_header) ;
    size_t blob_size = header->blob_size ;
    if (blob_size < sizeof (GB_blob_header))
    {
        // blob is invalid
        printf ("yikes! %d\n", __LINE__) ;
        return (GrB_PANIC) ;
    }

    // determine the matrix type
    GB_Type_code ccode = header->typecode ;
    GrB_Type ctype = GB_code_type (ccode, user_type) ;

    // ensure the type has the right size
    if (ctype == NULL || ctype->size != header->typesize)
    {
        // blob is invalid
        printf ("yikes! %d\n", __LINE__) ;
        return (GrB_PANIC) ;
    }

    // ensure the user-defined type has the right name
    if (ccode == GB_UDT_code)
    {
        if ((blob_size < sizeof (GB_blob_header) + GB_LEN) ||
            (strncmp (blob + s, ctype->name, GB_LEN) != 0))
        {
            // blob is invalid
            printf ("yikes! %d\n", __LINE__) ;
            return (GrB_PANIC) ;
        }
    }

    // allocate the matrix with info from the header
    int sparsity = header->sparsity ;
    GB_OK (GB_new (Chandle, false, ctype, header->vlen, header->vdim,
        GB_Ap_null, header->is_csc, sparsity, header->hyper_switch,
        header->nvec, Context)) ;

    // extract the remainder of the header
    C = (*Chandle) ;
    C->nvec = header->nvec ;
    C->nvec_nonempty = header->nvec_nonempty ;
    C->nvals = header->nvals ;
    C->bitmap_switch = header->bitmap_switch ;
    C->sparsity_control = header->sparsity_control ;
    C->iso = header->iso ;

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
            GB_OK (GB_deserialize_from_blob (&(C->p), &(C->p_size),
                blob, blob_size, &s, Context)) ;
            GB_OK (GB_deserialize_from_blob (&(C->h), &(C->h_size),
                blob, blob_size, &s, Context)) ;
            GB_OK (GB_deserialize_from_blob (&(C->i), &(C->i_size),
                blob, blob_size, &s, Context)) ;
            break ;

        case GxB_SPARSE :

            // decompress Cp and Ci
            GB_OK (GB_deserialize_from_blob (&(C->p), &(C->p_size),
                blob, blob_size, &s, Context)) ;
            GB_OK (GB_deserialize_from_blob (&(C->i), &(C->i_size),
                blob, blob_size, &s, Context)) ;
            break ;

        case GxB_BITMAP : 

            // decompress Cb
            GB_OK (GB_deserialize_from_blob (&(C->b), &(C->b_size),
                blob, blob_size, &s, Context)) ;
            break ;

        case GxB_FULL : 
            break ;
        default: ;
    }

    // decompress Cx
    GB_OK (GB_deserialize_from_blob (&(C->x), &(C->x_size),
        blob, blob_size, &s, Context));

    //--------------------------------------------------------------------------
    // finalize matrix return result
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C from deserialize", GB0) ;
    return (GrB_SUCCESS) ;
}

