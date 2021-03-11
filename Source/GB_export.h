//------------------------------------------------------------------------------
// GB_export.h: definitions for import/export
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EXPORT_H
#define GB_EXPORT_H
#include "GB_transpose.h"

GrB_Info GB_import      // import a matrix in any format
(
    GrB_Matrix *A,      // handle of matrix to create
    GrB_Type type,      // type of matrix to create
    GrB_Index vlen,     // vector length
    GrB_Index vdim,     // vector dimension
    bool is_sparse_vector,      // true if A is a sparse GrB_Vector

    // the 5 arrays:
    GrB_Index **Ap,     // pointers, for sparse and hypersparse formats.
                        // Ignored for bitmap and full formats.
                        // NULL for GxB_Vector_import_CSC.
    GrB_Index Ap_size,  // size of Ap in bytes

    GrB_Index **Ah,     // vector indices.
                        // Ignored for sparse, bitmap, and full formats.
    GrB_Index Ah_size,  // size of Ah in bytes

    int8_t **Ab,        // bitmap, for bitmap format only.
                        // Ignored for hyper, sparse, and full formats.  
    GrB_Index Ab_size,  // size of Ab in bytes

    GrB_Index **Ai,     // indices for hyper and
                        // sparse formats.  Ignored for bitmap and full.
    GrB_Index Ai_size,  // size of Ai in bytes

    void **Ax,          // values
                        // Ax and *Ax are ignored if Ax_size is zero.
    GrB_Index Ax_size,  // size of Ax in bytes

    // additional information for specific formats:
    GrB_Index nvals,    // # of entries for bitmap format, or for a vector
                        // in CSC format.
    bool jumbled,       // if true, sparse/hypersparse may be jumbled.
    GrB_Index nvec,     // size of Ah for hypersparse format.

    // information for all formats:
    int sparsity,       // hypersparse, sparse, bitmap, or full
    bool is_csc,        // if true then matrix is by-column, else by-row
    GB_Context Context
) ;

GrB_Info GB_export      // export a matrix in any format
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix to export
    GrB_Index *vlen,    // vector length
    GrB_Index *vdim,    // vector dimension
    bool is_sparse_vector,      // true if A is a sparse GrB_Vector

    // the 5 arrays:
    GrB_Index **Ap,     // pointers, size nvec+1 for hyper, vdim+1 for sparse,
                        // NULL if A is a sparse CSC GrB_Vector
    GrB_Index *Ap_size, // size of Ap in bytes

    GrB_Index **Ah,     // vector indices, size nvec for hyper
    GrB_Index *Ah_size, // size of Ah in bytes

    int8_t **Ab,        // bitmap, size nzmax
    GrB_Index *Ab_size, // size of Ab in bytes

    GrB_Index **Ai,     // indices, size nzmax
    GrB_Index *Ai_size, // size of Ai in bytes

    void **Ax,          // values, size nzmax
    GrB_Index *Ax_size, // size of Ax in bytes

    // additional information for specific formats:
    GrB_Index *nvals,   // # of entries for bitmap format.
    bool *jumbled,      // if true, sparse/hypersparse may be jumbled.
    GrB_Index *nvec,    // size of Ah for hypersparse format.

    // information for all formats:
    int *sparsity,      // hypersparse, sparse, bitmap, or full
    bool *is_csc,       // if true then matrix is by-column, else by-row
    GB_Context Context
) ;

#endif

