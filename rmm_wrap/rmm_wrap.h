//------------------------------------------------------------------------------
// rmm_wrap/rmm_wrap.h: include file for rmm_wrap
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef RMM_WRAP_H
#define RMM_WRAP_H

// get the definition of GB_MAX_NGPUS
#include "Source/include/GB_system.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// TODO describe the modes
typedef enum
{
    rmm_wrap_host = 0,
    rmm_wrap_host_pinned = 1,
    rmm_wrap_device = 2,
    rmm_wrap_managed = 3
}
RMM_MODE ;

// get id of currently selected device
int rmm_wrap_get_current_device (void) ;

// determine if RMM has been initialized
bool rmm_wrap_is_initialized (void) ;

// create an RMM resource
int rmm_wrap_initialize
(
    uint32_t device_id,
    RMM_MODE mode,
    size_t init_pool_memsize,
    size_t max_pool_memsize
) ;

// initialize rmm_wrap_contexts for each device in CUDA_VISIBLE_DEVICES
// (or single device_id 0 if not specified)
int rmm_wrap_initialize_all_same
(
    RMM_MODE mode,
    size_t init_pool_memsize,
    size_t max_pool_memsize
) ;

// destroy an RMM resource
void rmm_wrap_finalize (void) ;

// The two PMR-based allocate/deallocate signatures (C-style) (based on current device_id):
void *rmm_wrap_allocate (size_t *size) ;
void  rmm_wrap_deallocate (void *p, size_t size) ;

// The four malloc/calloc/realloc/free signatures (based on current device_id):
void *rmm_wrap_malloc (size_t size) ;
// void *rmm_wrap_calloc (size_t n, size_t size) ;      // not used
// void *rmm_wrap_realloc (void *p, size_t newsize) ;   // not used
void  rmm_wrap_free (void *p) ;

#ifdef __cplusplus
}
#endif
#endif

