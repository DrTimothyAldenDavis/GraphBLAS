//------------------------------------------------------------------------------
// GB_jitifyer.h: definitions for the CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_JITIFYER_H
#define GB_JITIFYER_H

#include <dlfcn.h>

//------------------------------------------------------------------------------
// list of jitifyed kernels
//------------------------------------------------------------------------------

#define GB_JIT_KERNEL_REDUCE 1
#define GB_JIT_KERNEL_MXM_DOT3 2
#define GB_JIT_KERNEL_MXM_SAX3 3
// ... etc FIXME: list them all here
// or make this an enum

//------------------------------------------------------------------------------
// GB_jitifyer_entry: an entry in the jitifyer hash table
//------------------------------------------------------------------------------

struct GB_jit_encoding_struct
{
    uint64_t code ;         // from GB_enumify_*
    uint32_t kcode ;        // which kernel
    uint32_t suffix_len ;   // length of the suffix (0 for builtin)
} ;

typedef struct GB_jit_encoding_struct GB_jit_encoding ;

struct GB_jit_entry_struct
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

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash (codes) ;
    GB_jit_encoding *encoding,
    char *suffix
) ;

bool GB_jitifyer_insert         // return true if successful, false if failure
(
    // input:
    uint64_t hash,              // hash for the problem
    GB_jit_encoding *encoding,  // primary encoding
    char *suffix,               // suffix for user-defined types/operators
    void *dl_handle,            // library handle from dlopen
    void *dl_function           // function handle from dlsym
) ;

uint64_t GB_jitifyer_hash_encoding
(
    GB_jit_encoding *encoding
) ;

uint64_t GB_jitifyer_hash
(
    const void *bytes,      // any string of bytes
    size_t nbytes           // # of bytes to hash
) ;

void GB_jitifyer_finalize (void) ;

// to query a library for its type and operator definitions
typedef const char *(*GB_jit_query_defn_func) (int k) ;

// to query a library for its type and operator definitions
typedef bool (*GB_jit_query_monoid_func)
(
    void *id,
    void *term,
    size_t id_size,
    size_t term_size
) ;

// to query a library for its version
typedef void (*GB_jit_query_version_func)
(
    int *version
) ;

bool GB_jitifyer_match_defn     // return true if definitions match
(
    // input:
    void *dl_query,             // query_defn function pointer
    int k,                      // compare current_defn with query_defn (k)
    const char *current_defn    // current definition (or NULL if not present)
) ;

bool GB_jitifyer_match_idterm   // return true if monoid id and term match
(
    void *dl_handle,            // dl_handle for the jit kernel library
    GrB_Monoid monoid           // current monoid to compare
) ;

bool GB_jitifyer_match_version
(
    void *dl_handle             // dl_handle for the jit kernel library
) ;

int GB_jitifyer_compile         // return result of system() call
(
    const char *kernel_name     // kernel to compile
) ;

#endif

