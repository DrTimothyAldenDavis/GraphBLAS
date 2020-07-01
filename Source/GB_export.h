//------------------------------------------------------------------------------
// GB_export.h: definitions for import/export
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_EXPORT_H
#define GB_EXPORT_H
#include "GB_transpose.h"

//------------------------------------------------------------------------------
// macros for import/export
//------------------------------------------------------------------------------

#define GB_IMPORT_CHECK                                         \
    GB_RETURN_IF_NULL (A) ;                                     \
    (*A) = NULL ;                                               \
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;                        \
    if (nrows > GxB_INDEX_MAX || ncols > GxB_INDEX_MAX ||       \
        nvals > GxB_INDEX_MAX)                                  \
    {                                                           \
        return (GrB_INVALID_VALUE) ;                            \
    }                                                           \
    /* get the descriptor */                                    \
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ;

#define GB_EXPORT_CHECK                                         \
    GB_RETURN_IF_NULL (A) ;                                     \
    GB_RETURN_IF_NULL_OR_FAULTY (*A) ;                          \
    ASSERT_MATRIX_OK (*A, "A to export", GB0) ;                 \
    GB_RETURN_IF_NULL (type) ;                                  \
    GB_RETURN_IF_NULL (nrows) ;                                 \
    GB_RETURN_IF_NULL (ncols) ;                                 \
    GB_RETURN_IF_NULL (nvals) ;                                 \
    GB_RETURN_IF_NULL (nonempty) ;                              \
    /* get the descriptor */                                    \
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6) ; \
    /* finish any pending work */                               \
    GB_MATRIX_WAIT (*A) ;                                       \
    /* export basic attributes */                               \
    (*type) = (*A)->type ;                                      \
    (*nrows) = GB_NROWS (*A) ;                                  \
    (*ncols) = GB_NCOLS (*A) ;                                  \
    (*nvals) = GB_NNZ (*A) ;

#endif

