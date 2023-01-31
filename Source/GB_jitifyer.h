//------------------------------------------------------------------------------
// GB_jitifyer.h: definitions for the CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_JITIFYER_H
#define GB_JITIFYER_H

#include "GB.h"

bool GB_jitifyer_expand (void) ;

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash (codes) ;
    uint64_t *codes         // array of size 6
) ;

bool GB_jitifyer_insert
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash (codes) ;
    uint64_t *codes,        // array of size 6
    void *dl_handle,
    void *dl_function
) ;

uint64_t GB_jitifyer_hash (uint64_t *codes) ;

#endif

