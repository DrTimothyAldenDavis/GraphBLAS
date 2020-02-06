//------------------------------------------------------------------------------
// GB_AxB_saxpy3_header.h: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Definitions for GB_AxB_saxpy3*.*

#ifndef GB_AXB_SAXPY3_HEADER_H
#define GB_AXB_SAXPY3_HEADER_H

//------------------------------------------------------------------------------
// free workspace for GB_AxB_saxpy3
//------------------------------------------------------------------------------

#undef  GB_FREE_INITIAL_WORK
#define GB_FREE_INITIAL_WORK ;

#undef  GB_FREE_TASKLIST_AND_HASH_TABLES
#define GB_FREE_TASKLIST_AND_HASH_TABLES                                    \
{                                                                           \
    GB_FREE_MEMORY (*(TaskList_handle), ntasks, sizeof (GB_saxpy3task_struct));\
    GB_FREE_MEMORY (Hi_all, Hi_size_total, sizeof (int64_t)) ;              \
    GB_FREE_MEMORY (Hf_all, Hf_size_total, sizeof (int64_t)) ;              \
    GB_FREE_MEMORY (Hx_all, Hx_size_total, 1) ;                             \
}

#undef  GB_FREE_WORK
#define GB_FREE_WORK                                                        \
{                                                                           \
    GB_FREE_INITIAL_WORK ;                                                  \
    GB_FREE_TASKLIST_AND_HASH_TABLES ;                                      \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                                         \
{                                                                           \
    GB_FREE_WORK ;                                                          \
    GB_MATRIX_FREE (Chandle) ;                                              \
}

#endif
