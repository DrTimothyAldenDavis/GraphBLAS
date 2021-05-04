//------------------------------------------------------------------------------
// GB_rmm.h: definitions for RMM
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_RMM_H
#define GB_RMM_H

//------------------------------------------------------------------------------
// memory management via the Rapids Memory Manager
//------------------------------------------------------------------------------

// FIXME: these are working placeholders that do not use RMM

static inline void *GB_rmm_alloc (void *rmm_resource, size_t *size)
{
    return (GB_Global_malloc_function (*size)) ;
}

static inline void GB_rmm_dealloc (void *rmm_resource, void **p, size_t size)
{
    GB_Global_free_function (p) ;
}

#endif
