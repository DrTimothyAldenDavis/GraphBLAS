//------------------------------------------------------------------------------
// GB_arena.h: utilities for arenas
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ARENA_H
#define GB_ARENA_H

GrB_Info GB_set_arena           // set arena of a block of memory
(
    // input/output:
    void **p_handle,            // block of memory to change
    uint64_t *p_mem_handle,     // memsize and arena of block of memory
    // input
    const int new_arena,        // arena to move to
    const uint64_t new_memsize, // new size of the block of memory
    const uint64_t n,           // # of bytes that must be copied
    const int nthreads          // max # of threads to use
) ;

GrB_Info GB_set_arenas          // modify all arenas of a matrix
(
    // input/output
    GrB_Matrix *Ahandle,        // handle of matrix to modify
    // input
    const int new_header_arena, // new arena for the header of A
    const int new_data_arena    // new arena for the data content of A
) ;

#endif

