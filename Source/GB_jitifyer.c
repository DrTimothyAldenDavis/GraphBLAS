//------------------------------------------------------------------------------
// GB_jitifyer.c: CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_jitifyer.h"

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

    bool builtin = (bool) (encoding->primary.suffix_len == 0) ;

    // look up the entry in the hash table
    for (int64_t k = hash & GB_jit_table_bits ; ; k++)
    {
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_handle == NULL)
        {
            // found an empty entry, so the entry is not in the table
            // FIXME: place a marker here as a placeholder, so other user
            // threads know that the jit kernel is currently being compiled...
            return (NULL) ;
        }
        else if (e->hash == hash &&
            (memcmp (e->encoding, encoding, sizeof (GB_jit_encoding)) == 0) &&
            (builtin || (strcmp (e->suffix, suffix) == 0)))
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

bool GB_jitifyer_insert
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash (codes) ;
    GB_jit_encoding *encoding,
    char *suffix,
    void *dl_handle,
    void *dl_function
)
{

    // FIXME: need to place this entire function in a critical section

    //  printf ("insert hash %016" PRIx64 " code %016" PRIx64 "\n", hash,
    //      encoding->primary.code) ;

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
        for (int64_t kold = 0 ; kold < GB_jit_table_size ; kold++)
        {
            if (GB_jit_table [kold].dl_handle != NULL)
            {
                // rehash the entry to the larger hash table
                uint64_t hash = GB_jit_table [kold].hash ;
                int64_t  knew = hash & new_bits ;
                new_table [knew] = GB_jit_table [kold] ;
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

    uint32_t suffix_len = encoding->primary.suffix_len ;
    bool builtin = (bool) (suffix_len == 0) ;

    for (int64_t k = hash & GB_jit_table_bits ; ; k++)
    {
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_handle == NULL)
        {
            printf ("insert into %ld\n", k) ;
            e->hash = hash ;
            memcpy (&(e->encoding), encoding, sizeof (GB_jit_encoding)) ;
            e->suffix = NULL ;
            e->suffix_size = 0 ;
            if (!builtin)
            {
                size_t siz ;
                e->suffix = GB_MALLOC (suffix_len+1, sizeof (char), &siz) ;
                if (e->suffix == NULL)
                {
                    return (false) ;
                }
                e->suffix_size = siz ;
            }
            e->dl_handle = dl_handle ;
            e->dl_function = dl_function ;
            GB_jit_table_populated++ ;
            return (true) ;
        }
        // otherwise, keep looking
    }
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
    return (XXH3_64bits ((const void *) encoding, sizeof (GB_jit_encoding)) ;
}

uint64_t GB_jitifyer_suffix_hash
(
    char *suffix,       // string with operator name and types
    uint32_t suffix_len // length of the string, not including terminating '\0'
)
{
    return (XXH3_64bits ((const void *) suffix, (size_t) suffix_len)) ;
}

