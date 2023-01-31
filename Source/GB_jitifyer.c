//------------------------------------------------------------------------------
// GB_jitifyer.c: CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_jitifyer.h"

//------------------------------------------------------------------------------
// GB_jitifyer_entry: an entry in the jitifyer hash table
//------------------------------------------------------------------------------

// Note that the name of the function is not needed for the hash table.  It is
// constructed when the dynamic library is compiled, or first loaded from the
// constructed lib*.so.  The name is then used to create the dl_function
// pointer, but it is not needed afterwards.

// mxm requires 6 words of codes, each 64 bits
//      0:  the specific kernel (dot3, saxpy3, etc).
//          could be done in fewer bits, but needs to be enumerated across
//          all JIT kernels (mxm, reduce, ewiseAdd, select, ...) so there
//          are dozens
//      1:  the semiring_code (62 bits), includes M->type
//      2:  semiring pointer (handles all user-defined methods)
//      3:  A->type pointer
//      4:  B->type pointer
//      5:  C->type pointer

// reduce requires 4 words of codes:
//      0:  the kernel
//      1:  the rcode (~30 bits)
//      2:  monoid pointer
//      3:  A->type pointer

// ewise methods require 5 words of codes:
//      0:  the kernel
//      1:  the ecode (~47 bits)
//      2:  op pointer
//      3:  A->type pointer
//      4:  B->type pointer

// As a result, the following hash entry takes 9*sizeof(uint64_t) = 72 bytes.
// The 

struct GB_jit_entry_struct
{
    uint64_t hash ;         // hash = XXH3_64bits (codes [0:5], 6*8)
    uint64_t codes [6] ;    // fully unique encoding of the problem; not all
                            // entries use all of this array.
    void *dl_handle ;       // handle from dlopen, to be passed to dlclose
    void *dl_function ;     // address of the function itself, from dlsym
} ;

typedef struct GB_jit_entry_struct GB_jitifyer_entry ;

//------------------------------------------------------------------------------
// GB_jitifyer_hash_table: a hash table of jitifyer entries
//------------------------------------------------------------------------------

// The hash table is static and shared by all threads of the user application.
// It is only visible inside this file.  It starts out empty (NULL).  Its size
// is either zero (at the beginning), or a power of two (of size 1024 or more).

#define GB_JITIFIER_INITIAL_SIZE 1024

static GB_jitifyer_entry *GB_jit_table = NULL ;
static int64_t  GB_jit_table_size = 0 ;  // always a power of 2
static uint64_t GB_jit_table_bits = 0 ;  // hash mask (0xFFFF if size is 2^16)
static size_t   GB_jit_table_allocated = 0 ;
static int64_t  GB_jit_table_populated = 0 ;

//------------------------------------------------------------------------------
// GB_jitifyer_expand:  create or expand the hash table
//------------------------------------------------------------------------------

bool GB_jitifyer_expand (void)
{

    // FIXME: need to place this entire function in a critical section

    if (GB_jit_table == NULL)
    {

        //----------------------------------------------------------------------
        // allocate the initial hash table
        //----------------------------------------------------------------------

        GB_jit_table = GB_CALLOC (GB_JITIFIER_INITIAL_SIZE,
                struct GB_jit_entry_struct, &GB_jit_table_allocated) ;
        printf ("GB jit allocated of size %ld\n", GB_jit_table_allocated) ;
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
        GB_jitifyer_entry *new_table = GB_CALLOC (new_size,
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

    return (true) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_lookup:  find a kernel in the hash table
//------------------------------------------------------------------------------

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash (codes) ;
    uint64_t *codes         // array of size 6
)
{

    // FIXME: need to place this entire function in a critical section

    if (GB_jit_table == NULL)
    {
        // no table yet so it isn't present
        return (NULL) ;
    }

    // look up the kernel in the hash table
    for (int64_t k = hash & GB_jit_table_bits ; ; k++)
    {
        if (GB_jit_table [k].dl_handle == NULL)
        {
            // found an empty entry, so the kernel is not in the table
            // FIXME: place a marker here as a placeholder, so other user
            // threads know that the kernel is currently being compiled...
            return (NULL) ;
        }
        else if (hash == GB_jit_table [k].hash &&
                (codes [0] == GB_jit_table [k].codes [0]) &&
                (codes [1] == GB_jit_table [k].codes [1]) &&
                (codes [2] == GB_jit_table [k].codes [2]) &&
                (codes [3] == GB_jit_table [k].codes [3]) &&
                (codes [4] == GB_jit_table [k].codes [4]) &&
                (codes [5] == GB_jit_table [k].codes [5]))
        {
            // found the right entry
            return (GB_jit_table [k].dl_function) ;
        }
        // otherwise, keep looking
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_insert:  insert a kernel in the hash table
//------------------------------------------------------------------------------

bool GB_jitifyer_insert
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash (codes) ;
    uint64_t *codes,        // array of size 6
    void *dl_handle,
    void *dl_function
)
{
    
    // ensure the hash table is large enough
    printf ("insert hash %016" PRIx64 " code %016" PRIx64 "\n", hash,
        codes [1]) ;

    if (!GB_jitifyer_expand ( )) return (false) ;

    printf ("expanded\n") ;

    // look up the kernel in the hash table
    for (int64_t k = hash & GB_jit_table_bits ; ; k++)
    {
        printf ("look in %ld: %p\n", k, GB_jit_table) ;
        if (GB_jit_table [k].dl_handle == NULL)
        {
            printf ("empty slot\n") ;
            GB_jit_table [k].hash = hash ;
            GB_jit_table [k].codes [0] = codes [0] ;
            GB_jit_table [k].codes [1] = codes [1] ;
            GB_jit_table [k].codes [2] = codes [2] ;
            GB_jit_table [k].codes [3] = codes [3] ;
            GB_jit_table [k].codes [4] = codes [4] ;
            GB_jit_table [k].codes [5] = codes [5] ;
            GB_jit_table [k].dl_handle = dl_handle ;
            GB_jit_table [k].dl_function = dl_function ;
            GB_jit_table_populated++ ;
            printf ("added \n") ;
            return (true) ;
        }
        // otherwise, keep looking
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_hash:  compute the hash from the 6-word codes array
//------------------------------------------------------------------------------

// xxHash uses switch statements with no default case.
#pragma GCC diagnostic ignored "-Wswitch-default"

#define XXH_INLINE_ALL
#define XXH_NO_STREAM
#include "xxhash.h"

uint64_t GB_jitifyer_hash (uint64_t *codes)
{
    return (XXH3_64bits (codes, 6 * sizeof (uint64_t))) ;
}

