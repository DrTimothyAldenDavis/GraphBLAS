//------------------------------------------------------------------------------
// GB_jitifyer.c: CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"
#include <dlfcn.h>

//------------------------------------------------------------------------------
// GB_jitifyer_hash_table: a hash table of jitifyer entries
//------------------------------------------------------------------------------

// The hash table is static and shared by all threads of the user application.
// It is only visible inside this file.  It starts out empty (NULL).  Its size
// is either zero (at the beginning), or a power of two (of size 1024 or more).

#define GB_JITIFIER_INITIAL_SIZE 1024

static GB_jit_entry *GB_jit_table = NULL ;
static size_t   GB_jit_table_allocated = 0 ;

static int64_t  GB_jit_table_size = 0 ;  // always a power of 2
static uint64_t GB_jit_table_bits = 0 ;  // hash mask (0xFFFF if size is 2^16)
static int64_t  GB_jit_table_populated = 0 ;

static char    *GB_jit_cache_path = NULL ;
static size_t   GB_jit_cache_path_allocated = 0 ;

static char    *GB_jit_source_path = NULL ;
static size_t   GB_jit_source_path_allocated = 0 ;

static char    *GB_jit_C_compiler = NULL ;
static size_t   GB_jit_C_compiler_allocated = 0 ;

static char    *GB_jit_C_flags = NULL ;
static size_t   GB_jit_C_flags_allocated = 0 ;

static char    *GB_jit_library_name = NULL ;
static size_t   GB_jit_library_name_allocated = 0 ;

static char    *GB_jit_source_name = NULL ;
static size_t   GB_jit_source_name_allocated = 0 ;

static char    *GB_jit_include = NULL ;
static size_t   GB_jit_include_allocated = 0 ;

static char    *GB_jit_command = NULL ;
static size_t   GB_jit_command_allocated = 0 ;

//------------------------------------------------------------------------------
// GB_jitifyer_free: free the JIT table and all the strings
//------------------------------------------------------------------------------

void GB_jit_free (void)
{ 
    GB_FREE (&GB_jit_table, GB_jit_table_allocated) ;
    GB_jit_table_allocated = 0 ;
    GB_FREE (&GB_jit_cache_path, GB_jit_cache_path_allocated) ;
    GB_jit_cache_path_allocated = 0 ;
    GB_FREE (&GB_jit_source_path, GB_jit_source_path_allocated) ;
    GB_jit_source_path_allocated = 0 ;
    GB_FREE (&GB_jit_C_compiler, GB_jit_C_compiler_allocated) ;
    GB_jit_C_compiler_allocated = 0 ;
    GB_FREE (&GB_jit_C_flags, GB_jit_C_flags_allocated) ;
    GB_jit_C_flags_allocated = 0 ;
    GB_FREE (&GB_jit_library_name, GB_jit_library_name_allocated) ;
    GB_jit_library_name_allocated = 0 ;
    GB_FREE (&GB_jit_source_name, GB_jit_source_name_allocated) ;
    GB_jit_source_name_allocated = 0 ;
    GB_FREE (&GB_jit_include, GB_jit_include_allocated) ;
    GB_jit_include_allocated = 0 ;
    GB_FREE (&GB_jit_command, GB_jit_command_allocated) ;
    GB_jit_command_allocated = 0 ;
    GB_jit_table_size = 0 ;
    GB_jit_table_bits = 0 ;
    GB_jit_table_populated = 0 ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_init: initialize the CPU and CUDA JIT folders, flags, etc
//------------------------------------------------------------------------------

#define OK(ok)                          \
    if (!(ok))                          \
    {                                   \
        GB_jit_free ( ) ;               \
        if (fp != NULL) fclose (fp) ;   \
        fp = NULL ;                     \
        GB_FREE (&str, str_alloc) ;     \
        return ;                        \
    }

void GB_jitifyer_init (void)
{
    size_t len = 0, str_alloc = 0 ;
    char *str = NULL ;
    FILE *fp = NULL ;

    //--------------------------------------------------------------------------
    // find the GB_jit_cache_path
    //--------------------------------------------------------------------------

    // printf ("JIT init:\n") ;
    char *cache_path = getenv ("GRAPHBLAS_CACHE_PATH") ;
    if (cache_path != NULL)
    { 
        // use the environment variable GRAPHBLAS_CACHE_PATH as-is
        len = strlen (cache_path) ;
        GB_jit_cache_path = GB_MALLOC (len+2, char,
            &(GB_jit_cache_path_allocated)) ;
        OK (GB_jit_cache_path != NULL) ;
        // printf ("cache %d %d\n", len, GB_jit_cache_path_allocated) ;
        strncpy (GB_jit_cache_path, cache_path, GB_jit_cache_path_allocated) ;
    }
    else
    { 
        // Linux, Mac, Unix: look for HOME
        cache_path = getenv ("HOME") ;
        char *dot = "." ;
        if (cache_path == NULL)
        { 
            // Windows: look for LOCALAPPDATA
            cache_path = getenv ("LOCALAPPDATA") ;
            dot = "" ;
        }
        if (cache_path != NULL)
        { 
            // found the cache_path
            size_t len = strlen (cache_path) + 80 ;
            GB_jit_cache_path = GB_MALLOC (len, char,
                &(GB_jit_cache_path_allocated)) ;
            OK (GB_jit_cache_path != NULL) ;
            // printf ("cache %d %d\n", len, GB_jit_cache_path_allocated) ;
            snprintf (GB_jit_cache_path,
                GB_jit_cache_path_allocated,
                "%s/%sSuiteSparse/GraphBLAS/%d.%d.%d", cache_path, dot,
                GxB_IMPLEMENTATION_MAJOR,
                GxB_IMPLEMENTATION_MINOR,
                GxB_IMPLEMENTATION_SUB) ;
        }
    }

    OK (GB_jit_cache_path != NULL) ;

    //--------------------------------------------------------------------------
    // open the GraphBLAS_config.txt file
    //--------------------------------------------------------------------------

    len = strlen (GB_jit_cache_path) + 80 ;
    str = GB_MALLOC (len, char, &str_alloc) ;
    OK (str != NULL) ;
    // printf ("str %d %d\n", len, str_alloc) ;

    snprintf (str, str_alloc, "%s/GraphBLAS_config.txt", GB_jit_cache_path) ;
    fp = fopen (str, "r") ;
    OK (fp != NULL) ;

    //--------------------------------------------------------------------------
    // determine the size of the GraphBLAS_config.txt file
    //--------------------------------------------------------------------------

    size_t file_size = 0 ;
    while (fgetc (fp) != EOF)
    { 
        file_size++ ;
    }
    rewind (fp) ;

    //--------------------------------------------------------------------------
    // reallocate workspace (large enough to hold the whole file)
    //--------------------------------------------------------------------------

    GB_FREE (&str, str_alloc) ;
    str = GB_MALLOC (file_size+2, char, &str_alloc) ;
    OK (str != NULL) ;
    // printf ("str %d %d\n", len, str_alloc) ;

    //--------------------------------------------------------------------------
    // parse the GraphBLAS_config.txt file
    //--------------------------------------------------------------------------

    // line 1: get the GB_jit_source_path
    OK (fgets (str, file_size+2, fp) != NULL) ;
    len = strlen (str) ;
    str [len-1] = '\0' ;
    GB_jit_source_path = GB_MALLOC (len + 2, char,
        &(GB_jit_source_path_allocated)) ;
    OK (GB_jit_source_path != NULL) ;
    // printf ("source %d %d\n", len, GB_jit_source_path_allocated) ;
    strncpy (GB_jit_source_path, str, GB_jit_source_path_allocated) ;

    // line 2: get the GB_jit_C_compiler
    OK (fgets (str, file_size+2, fp) != NULL) ;
    len = strlen (str) ;
    str [len-1] = '\0' ;
    GB_jit_C_compiler = GB_MALLOC (len + 2, char,
        &(GB_jit_C_compiler_allocated)) ;
    OK (GB_jit_C_compiler != NULL) ;
    // printf ("compiler %d %d\n", len, GB_jit_C_compiler_allocated) ;
    strncpy (GB_jit_C_compiler, str, GB_jit_C_compiler_allocated) ;

    // line 3: get the GB_jit_C_flags
    OK (fgets (str, file_size+2, fp) != NULL) ;
    len = strlen (str) ;
    str [len-1] = '\0' ;
    GB_jit_C_flags = GB_MALLOC (len + 2, char, &(GB_jit_C_flags_allocated)) ;
    OK (GB_jit_C_flags != NULL) ;
    // printf ("flags %d %d\n", len, GB_jit_C_flags_allocated) ;
    strncpy (GB_jit_C_flags, str, GB_jit_C_flags_allocated) ;

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    fclose (fp) ;
    fp = NULL ;
    GB_FREE (&str, str_alloc) ;

    //--------------------------------------------------------------------------
    // allocate permanent workspace
    //--------------------------------------------------------------------------

    len = GB_jit_cache_path_allocated + 300 + 2 * GxB_MAX_NAME_LEN ;
    GB_jit_library_name = GB_MALLOC (len, char,
        &(GB_jit_library_name_allocated)) ;
    OK (GB_jit_library_name != NULL) ;
    // printf ("libname %d %d\n", len, GB_jit_library_name_allocated) ;

    GB_jit_source_name = GB_MALLOC (len, char,
        &(GB_jit_source_name_allocated)) ;
    OK (GB_jit_source_name != NULL) ;
    // printf ("sourcename %d %d\n", len, GB_jit_source_name_allocated) ;

    len = 9 * GB_jit_source_path_allocated + 300 ;
    GB_jit_include = GB_MALLOC (len, char, &(GB_jit_include_allocated)) ;
    OK (GB_jit_include != NULL) ;
    // printf ("inc %d %d\n", len, GB_jit_include_allocated) ;

    snprintf (GB_jit_include, GB_jit_include_allocated,
        "-I%s/Include "
        "-I%s/Source "
        "-I%s/Source/Shared "
        "-I%s/Source/SharedTemplate "
        "-I%s/Source/Template "
        "-I%s/Source/JitKernels "
        "-I%s/cpu_features "
        "-I%s/cpu_features/include "
        "-I%s/GraphBLAS/rename ",
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path) ;

    size_t inc_len = strlen (GB_jit_include) ;
    len = 2 * GB_jit_C_flags_allocated + inc_len +
        4 * GB_jit_cache_path_allocated + 5 * GB_KLEN +
        GB_jit_source_path_allocated + 300 ;
    GB_jit_command = GB_MALLOC (len, char, &(GB_jit_command_allocated)) ;
    OK (GB_jit_command != NULL) ;
    // printf ("command len: %d\n", len) ;
    // printf ("cmd %d %d\n", len, GB_jit_command_allocated) ;
    // printf ("5 * GB_KLEN: %d\n", 5*GB_KLEN) ;

    printf ("cache:  %s\n", GB_jit_cache_path) ;
    printf ("source: [%s]\n", GB_jit_source_path) ;
    printf ("C compiler: [%s]\n", GB_jit_C_compiler) ;
    printf ("C flags: [%s]\n", GB_jit_C_flags) ;
    printf ("Include: [%s]\n", GB_jit_include) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_load: load a JIT kernel, compiling it if needed
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_load
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
)
{

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    if (hash == UINT64_MAX)
    { 
        return (GrB_NO_VALUE) ;
    }

    (*dl_function) = GB_jitifyer_lookup (hash, encoding, suffix) ;
    if ((*dl_function) != NULL)
    { 
        // found the kernel in the hash table
        GBURBLE ("(jit) ") ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // get the family properties
    //--------------------------------------------------------------------------

    GB_Operator op1 = NULL ;
    GB_Operator op2 = NULL ;
    char *family_name = NULL ;
    int scode_digits = 0 ;

    switch (family)
    {
        case GB_jit_apply_family  : 
            family_name = "apply" ;
            op1 = op ;
            scode_digits = 10 ;
            break ;

        case GB_jit_assign_family : 
            family_name = "assign" ;
            op1 = op ;
            scode_digits = 12 ;
            break ;

        case GB_jit_build_family  : 
            family_name = "build" ;
            op1 = op ;
            scode_digits = 7 ;
            break ;

        case GB_jit_ewise_family  : 
            family_name = "ewise" ;
            op1 = op ;
            scode_digits = 13 ;
            break ;

        case GB_jit_mxm_family    : 
            family_name = "mxm" ;
            monoid = semiring->add ;
            op1 = (GB_Operator) semiring->add->op ;
            op2 = (GB_Operator) semiring->multiply ;
            scode_digits = 16 ;
            break ;

        case GB_jit_reduce_family : 
            family_name = "reduce" ;
            op1 = (GB_Operator) monoid->op ;
            scode_digits = 7 ;
            break ;

        case GB_jit_select_family : 
            family_name = "select" ;
            op1 = op ;
            scode_digits = 10 ;
            break ;

        default: ;
    }

    //--------------------------------------------------------------------------
    // name the problem
    //--------------------------------------------------------------------------

    char kernel_name [GB_KLEN] ;
    GB_macrofy_name (kernel_name, "GB_jit", kname, scode_digits,
        encoding->code, suffix) ;

    //--------------------------------------------------------------------------
    // try to load the libkernel_name.so from the user's library folder
    //--------------------------------------------------------------------------

    snprintf (GB_jit_library_name, GB_jit_library_name_allocated,
        "%s/cpu/lib%s.so", GB_jit_cache_path, kernel_name) ;
    void *dl_handle = dlopen (GB_jit_library_name, RTLD_LAZY) ;

    //--------------------------------------------------------------------------
    // check if the kernel was found, but needs to be compiled anyway
    //--------------------------------------------------------------------------

    bool builtin = (encoding->suffix_len == 0) ;
    if (dl_handle != NULL && !builtin)
    { 
        // library is loaded but make sure the defn are OK
        void *dl_query = dlsym (dl_handle, "GB_jit_query_defn") ;
        bool need_to_compile = !GB_jitifyer_match_version (dl_handle) ||
        (op1 != NULL && !GB_jitifyer_match_defn (dl_query, 0, op1->defn)) ||
        (op2 != NULL && !GB_jitifyer_match_defn (dl_query, 1, op2->defn)) ||
        (type1 != NULL && !GB_jitifyer_match_defn (dl_query, 2, type1->defn)) ||
        (type2 != NULL && !GB_jitifyer_match_defn (dl_query, 3, type2->defn)) ||
        (type3 != NULL && !GB_jitifyer_match_defn (dl_query, 4, type3->defn)) ||
        (monoid != NULL && !GB_jitifyer_match_idterm (dl_handle, monoid)) ;
        if (need_to_compile)
        { 
            // library is loaded but needs to change, so close it
            dlclose (dl_handle) ;
            dl_handle = NULL ;
        }
    }

    //--------------------------------------------------------------------------
    // create and compile source file, if needed
    //--------------------------------------------------------------------------

    if (dl_handle == NULL)
    { 

        //----------------------------------------------------------------------
        // create the kernel source file
        //----------------------------------------------------------------------

        GBURBLE ("(jit compile and load) ") ;
        snprintf (GB_jit_source_name, GB_jit_source_name_allocated,
            "%s/cpu/%s.c", GB_jit_cache_path, kernel_name) ;
        FILE *fp = fopen (GB_jit_source_name, "w") ;
        if (fp == NULL)
        { 
            // FIXME: use another error code here
            printf ("cannot open source file: %s\n", GB_jit_source_name) ;
            return (GrB_PANIC) ;
        }
        fprintf (fp,
            "//--------------------------------------"
            "----------------------------------------\n"
            "// %s.c\n", kernel_name) ;
        GB_macrofy_copyright (fp) ;
        fprintf (fp, "#include \"GB_jit_kernel_%s.h\"\n\n", family_name) ;
        GB_macrofy_family (fp, family, encoding->code, semiring, monoid,
            op, type1, type2, type3) ;
        fprintf (fp, "\n#include \"GB_jit_kernel_%s.c\"\n\n", kname) ;
        if (!builtin)
        { 
            // create query_defn function
            GB_macrofy_query_defn (fp, op1, op2, type1, type2, type3) ;
        }
        if (monoid != NULL)
        { 
            // create query_monoid function if the monoid is not builtin
            GB_macrofy_query_monoid (fp, monoid) ;
        }
        GB_macrofy_query_version (fp) ;
        fclose (fp) ;

        //----------------------------------------------------------------------
        // compile the source file to create the lib*.so file
        //----------------------------------------------------------------------

        GB_jitifyer_compile (kernel_name) ;
        dl_handle = dlopen (GB_jit_library_name, RTLD_LAZY) ;
        if (dl_handle == NULL)
        { 
            // unable to open lib*.so file: punt to generic
            // FIXME: use another error code here
            printf ("cannot load library .so\n") ;
            return (GrB_PANIC) ;
        }
    }
    else
    { 
        GBURBLE ("(jit load) ") ;
    }

    //--------------------------------------------------------------------------
    // get the jit_kernel_function pointer
    //--------------------------------------------------------------------------

    (*dl_function) = dlsym (dl_handle, "GB_jit_kernel") ;
    if ((*dl_function) == NULL)
    { 
        // unable to find GB_jit_kernel: punt to generic
        dlclose (dl_handle) ; 
        printf ("cannot load kernel\n") ;
        return (GrB_PANIC) ;
    }

    // insert the new kernel into the hash table
    if (!GB_jitifyer_insert (hash, encoding, suffix, dl_handle,
        (*dl_function)))
    { 
        // unable to add kernel to hash table: punt to generic
        dlclose (dl_handle) ; 
        return (GrB_OUT_OF_MEMORY) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_lookup:  find a jit entry in the hash table
//------------------------------------------------------------------------------

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash_encoding (encoding) ;
    GB_jit_encoding *encoding,
    const char *suffix
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
    const char *suffix,         // suffix for user-defined types/operators
    void *dl_handle,            // library handle from dlopen
    void *dl_function           // function handle from dlsym
)
{

    // FIXME: need to place this entire function in a critical section
    // FIXME return GrB_Info instead of bool

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // the kernel must not already appear in the hash table
    ASSERT (GB_jitifyer_lookup (hash, encoding, suffix) == NULL) ;

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
                e->suffix = GB_MALLOC (suffix_len+1, char, &(e->suffix_size)) ;
                if (e->suffix == NULL)
                { 
                    // out of memory
                    return (false) ;
                }
                strncpy (e->suffix, suffix, suffix_len+1) ;
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

    // clear all entries in the table
    if (GB_jit_table == NULL)
    {
        for (int64_t k = 0 ; k < GB_jit_table_size ; k++)
        {
            GB_jit_entry *e = &(GB_jit_table [k]) ;
            if (e->dl_handle != NULL)
            {
                // found an entry; free the suffix if present
                if (e->suffix != NULL)
                { 
                    GB_FREE (&(e->suffix), e->suffix_size) ;
                }
                // unload the dl library
                dlclose (e->dl_handle) ;
            }
        }
    }

    // free the table and all workspace
    GB_jit_free ( ) ;
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
        // ensure the definition hasn't changed
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

int GB_jitifyer_compile (char *kernel_name)
{ 

    snprintf (GB_jit_command, GB_jit_command_allocated,
    "gcc -fPIC "
    #ifdef GBRENAME
    "-DGBRENAME=1 "
    #endif
    "%s "                       // C flags
    "%s "                       // include directories
    " -o %s/cpu/%s.o "          // *.o file, first gcc command
    " -c %s/cpu/%s.c ;"         // *.c file, first gcc command
    "gcc -fPIC "                // 2nd gcc command
    "%s -shared "               // C flags for 2nd gcc command
    " -Wl,-soname,lib%s.so "    // soname 
    " -o %s/cpu/lib%s.so"       // lib*.so output
    " %s/cpu/%s.o "             // *.o file for 2nd gcc commnand
    " %s%s/build/libgraphblas%s.so -lm "    // libgraphblas.so
    ,
    GB_jit_C_flags, GB_jit_include,
    GB_jit_cache_path, kernel_name,     // *.o file, first gcc command
    GB_jit_cache_path, kernel_name,     // *.c file, first gcc command
    GB_jit_C_flags,                     // C flags for 2nd gcc command
    kernel_name,                        // soname
    GB_jit_cache_path, kernel_name,     // lib*.so output file
    GB_jit_cache_path, kernel_name,     // *.o file for 2nd gcc
    GB_jit_source_path,                 // libgraphblas.so
    #ifdef GBRENAME
    "/GraphBLAS", "_matlab"
    #else
    "", ""
    #endif
    ) ;

    printf ("command: %s\n", GB_jit_command) ;

    // compile the library and return result
    int result = system (GB_jit_command) ;
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
// types to indicate that its hash value is not required.

// A hash value of UINT64_MAX is also special: it denotes an object that cannot
// be JIT'd.

// So in the nearly impossible case that XXH3_64bits returns a hash value that
// happens to be zero or UINT64_MAX, it is reset to GB_MAGIC instead.

uint64_t GB_jitifyer_hash_encoding
(
    GB_jit_encoding *encoding
)
{ 
    uint64_t hash ;
    hash = XXH3_64bits ((const void *) encoding, sizeof (GB_jit_encoding)) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

uint64_t GB_jitifyer_hash
(
    const void *bytes,      // any string of bytes
    size_t nbytes,          // # of bytes to hash
    bool jitable            // true if the object can be JIT'd
)
{ 
    if (bytes == NULL || nbytes == 0) return (0) ;
    if (!jitable) return (UINT64_MAX) ;
    uint64_t hash ;
    hash = XXH3_64bits (bytes, nbytes) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

