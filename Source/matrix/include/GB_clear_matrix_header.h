//------------------------------------------------------------------------------
// GB_clear_matrix_header.h: macros for allocating a new matrix header
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef  GBNSTATIC
#define GBNSTATIC 1

// FIXME: replace with GB_matrix_header_new and delete this file

#undef  GB_CLEAR_MATRIX_HEADER
#define GB_CLEAR_MATRIX_HEADER(XX,XX_header_handle)                         \
{                                                                           \
    uint64_t XX_mem = 0 ;   /* FIXME memlane */                             \
    XX = GB_CALLOC_MEMORY (1, sizeof (struct GB_Matrix_opaque), &XX_mem) ;  \
    if (XX != NULL)                                                         \
    {                                                                       \
        XX->header_mem = XX_mem ;                                           \
        XX->magic = GB_MAGIC2 ;                                             \
    }                                                                       \
}

