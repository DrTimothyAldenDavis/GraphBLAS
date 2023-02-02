//------------------------------------------------------------------------------
// GB_jitifyer.c: CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_jitifyer.h"
#include <dlfcn.h>

//------------------------------------------------------------------------------
// GB_jitifyer_hash_table: a hash table of jitifyer entries
//------------------------------------------------------------------------------

// The hash table is static and shared by all threads of the user application.
// It is only visible inside this file.  It starts out empty (NULL).  Its size
// is either zero (at the beginning), or a power of two (of size 1024 or more).

#define GB_JITIFIER_INITIAL_SIZE 1024

static GB_jit_entry *GB_jit_table = NULL ;
static int64_t  GB_jit_table_size = 0 ;  // always a power of 2
static uint64_t GB_jit_table_bits = 0 ;  // hash mask (0xFFFF if size is 2^16)
static size_t   GB_jit_table_allocated = 0 ;
static int64_t  GB_jit_table_populated = 0 ;

//------------------------------------------------------------------------------
// GB_jitifyer_lookup:  find a jit entry in the hash table
//------------------------------------------------------------------------------

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash_* (encoding) ;
    GB_jit_encoding *encoding,
    char *suffix            // ignored if builtin
)
{

    // FIXME: need to place this entire function in a critical section

    if (GB_jit_table == NULL)
    {
        // no table yet so it isn't present
        return (NULL) ;
    }

    uint32_t suffix_len = encoding->suffix_len ;
    bool builtin = (bool) (suffix_len == 0) ;

#if 0
    // dump the hash table
    for (int64_t k = 0 ; k < GB_jit_table_size ; k++)
    {
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_handle != NULL)
        {
            char *s = e->suffix ;
            printf ("k %4ld: \n", k) ;
            printf ("   hash %016" PRIx64 "\n", e->hash) ;
            printf ("   code %016" PRIx64 "\n", e->encoding.code) ;
            printf ("   kcode: %d\n", e->encoding.kcode) ;
            printf ("   suffix [%s]\n", (s == NULL) ? "" : s) ;
            printf ("   suffix_len %d \n", e->encoding.suffix_len) ;
            printf ("   handle %p\n", e->dl_handle) ;
            printf ("   func %p\n", e->dl_function) ;
        } 
    }
#endif

    // look up the entry in the hash table
    for (int64_t k = hash ; ; k++)
    {
        k = k & GB_jit_table_bits ;
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_handle == NULL)
        {
            // found an empty entry, so the entry is not in the table
            // FIXME: place a marker here as a placeholder, so other user
            // threads know that the jit kernel is currently being compiled...
            return (NULL) ;
        }
        else if (e->hash == hash &&
            e->encoding.code == encoding->code &&
            e->encoding.kcode == encoding->kcode &&
            e->encoding.suffix_len == suffix_len &&
            (builtin || (memcmp (e->suffix, suffix, suffix_len) == 0)))
        {
            // found the right entry: return the corresponding dl_function
            return (e->dl_function) ;
        }
        // otherwise, keep looking
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_insert:  insert a jit entry in the hash table
//------------------------------------------------------------------------------

bool GB_jitifyer_insert         // return true if successful, false if failure
(
    // input:
    uint64_t hash,              // hash for the problem
    GB_jit_encoding *encoding,  // primary encoding
    char *suffix,               // suffix for user-defined types/operators
    void *dl_handle,            // library handle from dlopen
    void *dl_function           // function handle from dlsym
)
{

    // FIXME: need to place this entire function in a critical section

    //--------------------------------------------------------------------------
    // ensure the hash table is large enough
    //--------------------------------------------------------------------------

    if (GB_jit_table == NULL)
    {

        //----------------------------------------------------------------------
        // allocate the initial hash table
        //----------------------------------------------------------------------

        GB_jit_table = GB_CALLOC (GB_JITIFIER_INITIAL_SIZE,
                struct GB_jit_entry_struct, &GB_jit_table_allocated) ;
        if (GB_jit_table == NULL)
        {
            // out of memory
            return (false) ;
        }
        GB_jit_table_size = GB_JITIFIER_INITIAL_SIZE ;
        GB_jit_table_bits = GB_JITIFIER_INITIAL_SIZE - 1 ; 

    }
    else if (GB_jit_table_populated >= GB_jit_table_size / 4)
    {

        //----------------------------------------------------------------------
        // expand the existing hash table by a factor of 4 and rehash
        //----------------------------------------------------------------------

        // create a new table that is four times the size
        int64_t new_size = 4 * GB_jit_table_size ;
        int64_t new_bits = new_size - 1 ;
        size_t  new_allocated ;
        GB_jit_entry *new_table = GB_CALLOC (new_size,
                struct GB_jit_entry_struct, &new_allocated) ;

        if (GB_jit_table == NULL)
        {
            // out of memory; leave the existing table as-is
            return (false) ;
        }

        // rehash into the new table
        for (int64_t k = 0 ; k < GB_jit_table_size ; k++)
        {
            if (GB_jit_table [k].dl_handle != NULL)
            {
                // rehash the entry to the larger hash table
                uint64_t hash = GB_jit_table [k].hash ;
                new_table [hash & new_bits] = GB_jit_table [k] ;
            }
        }

        // free the old table
        GB_FREE (&GB_jit_table, GB_jit_table_allocated) ;

        // use the new table
        GB_jit_table = new_table ;
        GB_jit_table_size = new_size ;
        GB_jit_table_bits = new_bits ;
        GB_jit_table_allocated = new_allocated ;
    }

    //--------------------------------------------------------------------------
    // insert the jit entry in the hash table
    //--------------------------------------------------------------------------

    uint32_t suffix_len = encoding->suffix_len ;
    bool builtin = (bool) (suffix_len == 0) ;

    for (int64_t k = hash ; ; k++)
    {
        k = k & GB_jit_table_bits ;
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_handle == NULL)
        {
            // found an empty slot
            e->suffix = NULL ;
            e->suffix_size = 0 ;
            if (!builtin)
            {
                // allocate the suffix if the kernel is not builtin
                size_t siz ;
                e->suffix = GB_MALLOC (suffix_len+1, char, &siz) ;
                if (e->suffix == NULL)
                {
                    // out of memory
                    return (false) ;
                }
                strncpy (e->suffix, suffix, suffix_len+1) ;
                e->suffix_size = siz ;
            }
            e->hash = hash ;
            memcpy (&(e->encoding), encoding, sizeof (GB_jit_encoding)) ;
            e->dl_handle = dl_handle ;
            e->dl_function = dl_function ;
            GB_jit_table_populated++ ;
            return (true) ;
        }
        // otherwise, keep looking
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_finalize:  free the hash and clear all loaded kernels
//------------------------------------------------------------------------------

void GB_jitifyer_finalize (void)
{

    if (GB_jit_table == NULL)
    {
        // no table yet so nothing to do
        return ;
    }

    // clear all entries in the table
    for (int64_t k = 0 ; k < GB_jit_table_size ; k++)
    {
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_handle != NULL)
        {
            // found an entry; free the suffix if present
            GB_FREE (&(e->suffix), e->suffix_size) ;
            // unload the dl library
            dlclose (e->dl_handle) ;
        }
    }

    // free the table
    GB_FREE (&GB_jit_table, GB_jit_table_allocated) ;
    GB_jit_table_allocated = 0 ;
    GB_jit_table_size = 0 ;
    GB_jit_table_bits = 0 ;
    GB_jit_table_allocated = 0 ;
    GB_jit_table_populated = 0 ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_hash:  compute the hash
//------------------------------------------------------------------------------

// xxHash uses switch statements with no default case.
#pragma GCC diagnostic ignored "-Wswitch-default"

#define XXH_INLINE_ALL
#define XXH_NO_STREAM
#include "xxhash.h"

uint64_t GB_jitifyer_encoding_hash
(
    GB_jit_encoding *encoding
)
{
    return (XXH3_64bits ((const void *) encoding, sizeof (GB_jit_encoding))) ;
}

uint64_t GB_jitifyer_suffix_hash
(
    char *suffix,       // string with operator name and types
    uint32_t suffix_len // length of the string, not including terminating '\0'
)
{
    return (XXH3_64bits ((const void *) suffix, (size_t) suffix_len)) ;
}

