//------------------------------------------------------------------------------
// GB_export.h: definitions for import/export
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

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
    GrB_Index nzmax,    // size of Ai and Ax for sparse/hypersparse
    GrB_Index nvals,    // # of entries for bitmap
    bool jumbled,       // if true, sparse/hypersparse may be jumbled
    int64_t nonempty,   // # of non-empty vectors for sparse/hypersparse
    GrB_Index nvec,     // size of Ah for hypersparse
    GrB_Index **Ap,     // pointers, size nvec+1 for hyper, vdim+1 for sparse
    GrB_Index **Ah,     // vector indices, size nvec for hyper
    int8_t **Ab,        // bitmap, size nzmax
    GrB_Index **Ai,     // indices, size nzmax
    void **Ax,          // values, size nzmax
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
    GrB_Index *nzmax,   // size of Ab, Ai, and Ax
    GrB_Index *nvals,   // # of entries for bitmap matrices
    bool *jumbled,      // if true, sparse/hypersparse may be jumbled
    int64_t *nonempty,  // # of non-empty vectors for sparse/hypersparse
    GrB_Index *nvec,    // size of Ah for hypersparse
    GrB_Index **Ap,     // pointers, size nvec+1 for hyper, vdim+1 for sparse
    GrB_Index **Ah,     // vector indices, size nvec for hyper
    int8_t **Ab,        // bitmap, size nzmax
    GrB_Index **Ai,     // indices, size nzmax
    void **Ax,          // values, size nzmax
    int *sparsity,      // hypersparse, sparse, bitmap, or full
    bool *is_csc,       // if true then export matrix by-column, else by-row
    GB_Context Context
) ;

#endif

