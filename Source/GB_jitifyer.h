//------------------------------------------------------------------------------
// GB_jitifyer.h: definitions for the CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_JITIFYER_H
#define GB_JITIFYER_H

//------------------------------------------------------------------------------
// GB_jitifyer_entry: an entry in the jitifyer hash table
//------------------------------------------------------------------------------

struct GB_jit_encoding_struct   // 2*8 = 16 bytes
{
    uint64_t code ;         // from GB_enumify_*
    uint32_t kcode ;        // which kernel
    uint32_t suffix_len ;   // length of the suffix (0 for builtin)
} ;

typedef struct GB_jit_encoding_struct GB_jit_encoding ;

struct GB_jit_entry_struct      // 6*8 = 48 bytes
{
    uint64_t hash ;             // hash code for the problem
    GB_jit_encoding encoding ;  // encoding of the problem, except for suffix
    char *suffix ;              // kernel suffix for user-defined op / types,
                                // NULL for built-in kernels
    size_t suffix_size ;        // size of suffix malloc'd block
    void *dl_handle ;           // handle from dlopen, to be passed to dlclose
    void *dl_function ;         // address of the function itself, from dlsym
} ;

typedef struct GB_jit_entry_struct GB_jit_entry ;

//------------------------------------------------------------------------------
// GB_jitifyer methods for GraphBLAS
//------------------------------------------------------------------------------

bool GB_jitifyer_expand (void) ;

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash (codes) ;
    GB_jit_encoding *encoding,
    char *suffix
) ;

bool GB_jitifyer_insert
(
    // input:
    uint64_t hash,          // hash of the problem
    GB_jit_encoding *encoding,  // primary encoding
    char *suffix,           // suffix for user-defined types/operators
    void *dl_handle,
    void *dl_function
) ;

uint64_t GB_jitifyer_encoding_hash
(
    GB_jit_encoding *encoding
) ;

uint64_t GB_jitifyer_suffix_hash
(
    char *suffix,       // string with operator name and types
    uint32_t suffix_len // length of the string, not including terminating '\0'
) ;


#endif

