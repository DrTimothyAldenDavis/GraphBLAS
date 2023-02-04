//------------------------------------------------------------------------------
// GB_jitifyer.c: CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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
    printf ("\nlookup GB_jit_table at %p\n", GB_jit_table) ;
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
            // printf ("hash table [%ld] insert\n", k) ;
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
                // printf ("   suffix %p\n", e->suffix) ;
                strncpy (e->suffix, suffix, suffix_len+1) ;
                // printf ("       [%s]\n", e->suffix) ;
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

    // printf ("GB_jit_table at %p\n", GB_jit_table) ;

    // clear all entries in the table
    for (int64_t k = 0 ; k < GB_jit_table_size ; k++)
    {
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_handle != NULL)
        {
            // found an entry; free the suffix if present
            // printf ("free hash table [%ld]: %p\n", k, e->dl_handle) ;
            if (e->suffix != NULL)
            {
                // printf ("    free the suffix: %p\n", e->suffix) ;
                GB_FREE (&(e->suffix), e->suffix_size) ;
            }
            // unload the dl library
            dlclose (e->dl_handle) ;
        }
    }

    // free the table
    // printf ("free the table %p\n", GB_jit_table) ;
    GB_FREE (&GB_jit_table, GB_jit_table_allocated) ;
    GB_jit_table_allocated = 0 ;
    GB_jit_table_size = 0 ;
    GB_jit_table_bits = 0 ;
    GB_jit_table_allocated = 0 ;
    GB_jit_table_populated = 0 ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_match_defn: check if library and current definitions match
//------------------------------------------------------------------------------

bool GB_jitifyer_match_defn     // return true if definitions match
(
    // input:
    void *dl_query,             // query_defn function pointer
    int k,                      // compare current_defn with query_defn (k)
    const char *current_defn    // current definition (or NULL if not present)
)
{
    if (dl_query == NULL)
    {
        // library is missing the query_defn method
        return (false) ;
    }
    GB_jit_query_defn_func query_defn = (GB_jit_query_defn_func) dl_query ;
    const char *library_defn = query_defn (k) ;
    if ((current_defn != NULL) != (library_defn != NULL))
    {
        // one is not NULL but the other is NULL
        return (false) ;
    }
    else if (current_defn != NULL)
    {
        // both definitions are present
        // ensure the defintion hasn't changed
        return (strcmp (library_defn, current_defn) == 0) ;
    }
    else
    {
        // both definitions are NULL, so they match
        return (true) ;
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_match_idterm: check if monoid identity and terminal values match
//------------------------------------------------------------------------------

bool GB_jitifyer_match_idterm   // return true if monoid id and term match
(
    void *dl_handle,            // dl_handle for the jit kernel library
    GrB_Monoid monoid           // current monoid to compare
)
{
    // compare the identity and terminal
    void *dl_query = dlsym (dl_handle, "GB_jit_query_monoid") ;
    if (dl_query == NULL)
    {
        // the library is invalid; need recompile it
        return (false) ;
    }
    // check the identity and terminal values
    GB_jit_query_monoid_func query_monoid = (GB_jit_query_monoid_func) dl_query;
    size_t zsize = monoid->op->ztype->size ;
    size_t tsize = (monoid->terminal == NULL) ? 0 : zsize ;
    return (query_monoid (monoid->identity, monoid->terminal, zsize, tsize)) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_match_version: check the version of a kernel
//------------------------------------------------------------------------------

bool GB_jitifyer_match_version
(
    void *dl_handle             // dl_handle for the jit kernel library
)
{
    // compare the version
    void *dl_query = dlsym (dl_handle, "GB_jit_query_version") ;
    if (dl_query == NULL)
    {
        // the library is invalid; need recompile it
        return (false) ;
    }
    // check the version
    int version [3] ;
    GB_jit_query_version_func query_version =
        (GB_jit_query_version_func) dl_query ;
    query_version (version) ;
    // return true if the version matches
    return ((version [0] == GxB_IMPLEMENTATION_MAJOR) &&
            (version [1] == GxB_IMPLEMENTATION_MINOR) &&
            (version [2] == GxB_IMPLEMENTATION_SUB)) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_compile: compile a kernel
//------------------------------------------------------------------------------

int GB_jitifyer_compile
(
    const char *kernel_name
)
{
    printf ("compiling %s\n", kernel_name) ;

    // FIXME: create this at GrB_init time
    char root_folder [256] ;
    snprintf (root_folder, 256, "%s",
    "/home/faculty/d/davis/cuda/GraphBLAS") ;

    // FIXME: create this at GrB_init time, or by GxB_set
    char lib_folder [2048] ;
    snprintf (lib_folder, 2047,
            "/home/faculty/d/davis/.SuiteSparse/GraphBLAS/v%d.%d.%d"
            #ifdef GBRENAME
            "_matlab"
            #endif
            ,
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB) ;

    // FIXME: create this at GrB_init time
    char include_files [4096] ;
    snprintf (include_files, 4096,
        "-I%s "
        "-I%s/Include "
        "-I%s/Source "
        "-I%s/Source/Shared "
        "-I%s/Source/SharedTemplate "
        "-I%s/Source/Template "
        "-I%s/Source/JitKernels "
        "-I%s/cpu_features "
        "-I%s/cpu_features/include "
        #ifdef GBRENAME
        "-I%s/GraphBLAS/rename "
        #endif
        ,
        lib_folder,
        root_folder,
        root_folder,
        root_folder,
        root_folder,
        root_folder,
        root_folder,
        root_folder,
        root_folder
        #ifdef GBRENAME
        , root_folder
        #endif
        ) ;

    char command [4096] ;

    // FIXME: allow user to set compiler and flags
    snprintf (command, 4096,
    "gcc -fPIC -O3 -std=c11 -fexcess-precision=fast "
    #ifdef GBRENAME
    " -DGBRENAME=1 "
    #endif
    "-fcx-limited-range -fno-math-errno -fwrapv -DNDEBUG "
    "-fopenmp %s -o "
    " %s/%s.o -c %s/%s.c ;" 
    "gcc -fPIC -O3 -std=c11 -fexcess-precision=fast "
    "-fcx-limited-range -fno-math-errno -fwrapv -DNDEBUG "
    "-fopenmp "
    " -shared -Wl,-soname,lib%s.so -o %s/lib%s.so"
    " %s/%s.o "
    " %s%s/build/libgraphblas%s.so -lm "
    ,
    include_files,
    lib_folder, kernel_name,    // *.o file, first gcc command
    lib_folder, kernel_name,    // *.c file, first gcc command
    kernel_name,                // soname
    lib_folder, kernel_name,    // lib*.so output file
    lib_folder, kernel_name,    // *.o file for 2nd gcc
    root_folder,
    #ifdef GBRENAME
    "/GraphBLAS", "_matlab"
    #else
    "", ""
    #endif
    ) ;

    printf ("command: %s\n", command) ;

    // compile the library and return result
    int result = system (command) ;
    return (result) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_hash:  compute the hash
//------------------------------------------------------------------------------

// xxHash uses switch statements with no default case.
#pragma GCC diagnostic ignored "-Wswitch-default"

#define XXH_INLINE_ALL
#define XXH_NO_STREAM
#include "xxhash.h"

// A hash value of zero is unique, and is used for all builtin operators and
// types to indicate that its hash value is not required.  So in the nearly
// impossible case that XXH3_64bits returns a hash value that happens to be
// zero, it is reset to GB_MAGIC instead.

uint64_t GB_jitifyer_hash_encoding
(
    GB_jit_encoding *encoding
)
{
    uint64_t hash ;
    hash = XXH3_64bits ((const void *) encoding, sizeof (GB_jit_encoding)) ;
    return ((hash == 0) ? GB_MAGIC : hash) ;
}

uint64_t GB_jitifyer_hash
(
    const void *bytes,      // any string of bytes
    size_t nbytes           // # of bytes to hash
)
{
    if (bytes == NULL || nbytes == 0) return (0) ;
    uint64_t hash ;
    hash = XXH3_64bits (bytes, nbytes) ;
    return ((hash == 0) ? GB_MAGIC : hash) ;
}

