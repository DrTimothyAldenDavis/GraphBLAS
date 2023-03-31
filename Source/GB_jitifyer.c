//------------------------------------------------------------------------------
// GB_jitifyer.c: CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"
#include "GB_config.h"

// FIXME: dlfcn.h only exists on Linux/Unix/Mac; need to port to Windows
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

static char    *GB_jit_C_link_flags = NULL ;
static size_t   GB_jit_C_link_flags_allocated = 0 ;

static char    *GB_jit_library_name = NULL ;
static size_t   GB_jit_library_name_allocated = 0 ;

static char    *GB_jit_kernel_name = NULL ;
static size_t   GB_jit_kernel_name_allocated = 0 ;

static char    *GB_jit_include = NULL ;
static size_t   GB_jit_include_allocated = 0 ;

static char    *GB_jit_command = NULL ;
static size_t   GB_jit_command_allocated = 0 ;

static GxB_JIT_Control GB_jit_control =
    #if GB_JIT_ENABLED
    GxB_JIT_ON ;        // JIT enabled
    #else
    GxB_JIT_NONE ;      // JIT disabled at compile time
    #endif

//------------------------------------------------------------------------------
// GB_jitifyer_finalize: free the JIT table and all the strings
//------------------------------------------------------------------------------

#define OK(ok)                          \
    if (!(ok))                          \
    {                                   \
        GB_jitifyer_finalize ( ) ;      \
        return (GrB_OUT_OF_MEMORY) ;    \
    }

#define GB_FREE_STUFF(X)                \
{                                       \
    GB_FREE (&X, X ## _allocated) ;     \
    X ## _allocated = 0 ;               \
}

#define GB_MALLOC_STUFF(X,len)                          \
{                                                       \
    X = GB_MALLOC (len + 2, char, &(X ## _allocated)) ; \
    OK (X != NULL) ;                                    \
}

#define GB_COPY_STUFF(X,src)                            \
{                                                       \
    size_t len = strlen (src) ;                         \
    GB_MALLOC_STUFF (X, len) ;                          \
    strncpy (X, src, X ## _allocated) ;                 \
}

void GB_jitifyer_finalize (void)
{ 
    GB_jitifyer_table_free ( ) ;
    GB_FREE_STUFF (GB_jit_cache_path) ;
    GB_FREE_STUFF (GB_jit_source_path) ;
    GB_FREE_STUFF (GB_jit_C_compiler) ;
    GB_FREE_STUFF (GB_jit_C_flags) ;
    GB_FREE_STUFF (GB_jit_C_link_flags) ;
    GB_FREE_STUFF (GB_jit_library_name) ;
    GB_FREE_STUFF (GB_jit_kernel_name) ;
    GB_FREE_STUFF (GB_jit_include) ;
    GB_FREE_STUFF (GB_jit_command) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_init: initialize the CPU and CUDA JIT folders, flags, etc
//------------------------------------------------------------------------------

// Returns GrB_SUCCESS, GrB_OUT_OF_MEMORY, or GrB_NO_VALUE if the cache path
// cannot be found.

GrB_Info GB_jitifyer_init (void)
{

    //--------------------------------------------------------------------------
    // find the GB_jit_cache_path
    //--------------------------------------------------------------------------

    char *cache_path = getenv ("GRAPHBLAS_CACHE_PATH") ;
    if (cache_path != NULL)
    { 
        // use the environment variable GRAPHBLAS_CACHE_PATH as-is
        GB_COPY_STUFF (GB_jit_cache_path, cache_path) ;
    }
    else
    { 
        // Linux, Mac, Unix: look for HOME
        char *home = getenv ("HOME") ;
        char *dot = "." ;
        if (home == NULL)
        { 
            // Windows: look for LOCALAPPDATA
            home = getenv ("LOCALAPPDATA") ;
            dot = "" ;
        }
        if (home != NULL)
        { 
            // found home; create the cache path
            size_t len = strlen (home) + 60 ;
            GB_MALLOC_STUFF (GB_jit_cache_path, len) ;
            snprintf (GB_jit_cache_path,
                GB_jit_cache_path_allocated,
                "%s/%sSuiteSparse/GraphBLAS/%d.%d.%d", home, dot,
                GxB_IMPLEMENTATION_MAJOR,
                GxB_IMPLEMENTATION_MINOR,
                GxB_IMPLEMENTATION_SUB) ;
        }
    }

    if (GB_jit_cache_path == NULL)
    { 
        // cannot determine the JIT cache path; use the cmake build directory
        GB_COPY_STUFF (GB_jit_cache_path, GB_BUILD_PATH) ;
    }

    //--------------------------------------------------------------------------
    // initialize the remaining strings
    //--------------------------------------------------------------------------

    GB_COPY_STUFF (GB_jit_C_compiler,   GB_C_COMPILER) ;
    GB_COPY_STUFF (GB_jit_C_flags,      GB_C_FLAGS) ;
    GB_COPY_STUFF (GB_jit_C_link_flags, GB_C_LINK_FLAGS) ;
    GB_COPY_STUFF (GB_jit_source_path,  GB_SOURCE_PATH) ;

    //--------------------------------------------------------------------------
    // set the include string and allocate permanent workspace
    //--------------------------------------------------------------------------

    OK (GB_jitifyer_include ( ) == GrB_SUCCESS) ;
    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_control: get the JIT control
//------------------------------------------------------------------------------

GxB_JIT_Control GB_jitifyer_get_control (void)
{
    GxB_JIT_Control control ;
    #pragma omp critical (GB_jitifyer_worker)
    {
        control = GB_jit_control ;
    }
    return (control) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_control: set the JIT control
//------------------------------------------------------------------------------

void GB_jitifyer_set_control (int control)
{
    #if GB_JIT_ENABLED
    #pragma omp critical (GB_jitifyer_worker)
    {
        control = GB_IMAX (control, GxB_JIT_OFF) ;
        control = GB_IMIN (control, GxB_JIT_ON) ;
        GB_jit_control = (GxB_JIT_Control) control ;
        if (GB_jit_control == GxB_JIT_OFF)
        {
            // free all loaded JIT kernels and free the JIT hash table
            GB_jitifyer_table_free ( ) ;
        }
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_jitifyer_include:  allocate and determine -Istring
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_include (void)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_jit_source_path != NULL) ;

    //--------------------------------------------------------------------------
    // allocate and determine GB_jit_include
    //--------------------------------------------------------------------------

    size_t len = 10 * GB_jit_source_path_allocated + 200 ;
    GB_MALLOC_STUFF (GB_jit_include, len) ;

    snprintf (GB_jit_include, GB_jit_include_allocated,
        "-I%s/Include "
        "-I%s/Source "
        "-I%s/Source/Shared "
        "-I%s/Source/SharedTemplate "
        "-I%s/Source/Template "
        "-I%s/Source/JitKernels "
        "-I%s/rmm_wrap "
        "-I%s/cpu_features "
        "-I%s/cpu_features/include "
        #ifdef GBRENAME
        "-I%s/GraphBLAS/rename "
        #endif
        ,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path,
        GB_jit_source_path
        #ifdef GBRENAME
        , GB_jit_source_path
        #endif
        ) ;

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_alloc_space: allocate workspaces for the JIT
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_alloc_space (void)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (GB_jit_C_flags == NULL ||
        GB_jit_include == NULL ||
        GB_jit_C_link_flags == NULL ||
        GB_jit_cache_path == NULL ||
        GB_jit_source_path == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // allocate GB_jit_kernel_name if needed
    //--------------------------------------------------------------------------

    if (GB_jit_kernel_name == NULL)
    {
        size_t len = GB_jit_cache_path_allocated + 300 + 2 * GxB_MAX_NAME_LEN ;
        GB_MALLOC_STUFF (GB_jit_kernel_name, len) ;
    }

    //--------------------------------------------------------------------------
    // allocate GB_jit_library_name if needed
    //--------------------------------------------------------------------------

    if (GB_jit_library_name == NULL)
    {
        size_t len = GB_jit_cache_path_allocated + 300 + 2 * GxB_MAX_NAME_LEN ;
        GB_MALLOC_STUFF (GB_jit_library_name, len) ;
    }

    //--------------------------------------------------------------------------
    // allocate GB_jit_command if needed
    //--------------------------------------------------------------------------

    if (GB_jit_command == NULL)
    {
        size_t len = 2 * GB_jit_C_compiler_allocated +
            2 * GB_jit_C_flags_allocated + strlen (GB_jit_include) +
            4 * GB_jit_cache_path_allocated + 5 * GB_KLEN +
            GB_jit_source_path_allocated + 300 ;
        GB_MALLOC_STUFF (GB_jit_command, len) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_source_path: return the current source path
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_source_path (void)
{
    const char *s ;
    #pragma omp critical (GB_jitifyer_worker)
    {
        s = GB_jit_source_path ;
    }
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_source_path: set source path
//------------------------------------------------------------------------------

// Redefines the GB_jit_source_path.  This requires the -Istring to
// reconstructed, the command buffer to be reallocated.

GrB_Info GB_jitifyer_set_source_path (const char *new_source_path)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_source_path == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the source path in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    #pragma omp critical (GB_jitifyer_worker)
    { 
        info = GB_jitifyer_set_source_path_worker (new_source_path) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_source_path_worker: set source path in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_source_path_worker (const char *new_source_path)
{

    //--------------------------------------------------------------------------
    // free the old strings that depend on the source path
    //--------------------------------------------------------------------------

    GB_FREE_STUFF (GB_jit_source_path) ;
    GB_FREE_STUFF (GB_jit_include) ;
    GB_FREE_STUFF (GB_jit_command) ;

    //--------------------------------------------------------------------------
    // allocate the new GB_jit_source_path
    //--------------------------------------------------------------------------

    GB_COPY_STUFF (GB_jit_source_path, new_source_path) ;

    //--------------------------------------------------------------------------
    // allocate and define strings that depend on GB_jit_source_path
    //--------------------------------------------------------------------------

    OK (GB_jitifyer_include ( ) == GrB_SUCCESS) ;
    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_cache_path: return the current cache path
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_cache_path (void)
{
    const char *s ;
    #pragma omp critical (GB_jitifyer_worker)
    {
        s = GB_jit_cache_path ;
    }
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_cache_path: set a new cache path
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_cache_path (const char *new_cache_path)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_cache_path == NULL)
    {
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the cache path in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    #pragma omp critical (GB_jitifyer_worker)
    { 
        info = GB_jitifyer_set_cache_path_worker (new_cache_path) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_cache_path_worker: set cache path in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_cache_path_worker (const char *new_cache_path)
{

    //--------------------------------------------------------------------------
    // free the old strings that depend on the cache path
    //--------------------------------------------------------------------------

    GB_FREE_STUFF (GB_jit_cache_path) ;
    GB_FREE_STUFF (GB_jit_kernel_name) ;
    GB_FREE_STUFF (GB_jit_library_name) ;
    GB_FREE_STUFF (GB_jit_command) ;

    //--------------------------------------------------------------------------
    // allocate the new GB_jit_cache_path
    //--------------------------------------------------------------------------

    GB_COPY_STUFF (GB_jit_cache_path, new_cache_path) ;

    //--------------------------------------------------------------------------
    // allocate and define strings that depend on GB_jit_cache_path
    //--------------------------------------------------------------------------

    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_compiler: return the current C compiler
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_compiler (void)
{
    const char *s ;
    #pragma omp critical (GB_jitifyer_worker)
    {
        s = GB_jit_C_compiler ;
    }
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_compiler: set a new C compiler
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_compiler (const char *new_C_compiler)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_C_compiler == NULL)
    {
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C compiler in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    #pragma omp critical (GB_jitifyer_worker)
    { 
        info = GB_jitifyer_set_C_compiler_worker (new_C_compiler) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_compiler_worker: set C compiler in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_compiler_worker (const char *new_C_compiler)
{

    //--------------------------------------------------------------------------
    // free the old strings that depend on the C compiler
    //--------------------------------------------------------------------------

    GB_FREE_STUFF (GB_jit_C_compiler) ;
    GB_FREE_STUFF (GB_jit_command) ;

    //--------------------------------------------------------------------------
    // allocate the new GB_jit_C_compiler
    //--------------------------------------------------------------------------

    GB_COPY_STUFF (GB_jit_C_compiler, new_C_compiler) ;

    //--------------------------------------------------------------------------
    // allocate and define strings that depend on GB_jit_C_compiler
    //--------------------------------------------------------------------------

    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_flags: return the current C flags
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_flags (void)
{
    const char *s ;
    #pragma omp critical (GB_jitifyer_worker)
    {
        s = GB_jit_C_flags ;
    }
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_flags: set new C flags
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_flags (const char *new_C_flags)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_C_flags == NULL)
    {
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C flags in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    #pragma omp critical (GB_jitifyer_worker)
    { 
        info = GB_jitifyer_set_C_flags_worker (new_C_flags) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_flags_worker: set C flags in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_flags_worker (const char *new_C_flags)
{

    //--------------------------------------------------------------------------
    // free the old strings that depend on the C flags
    //--------------------------------------------------------------------------

    GB_FREE_STUFF (GB_jit_C_flags) ;
    GB_FREE_STUFF (GB_jit_command) ;

    //--------------------------------------------------------------------------
    // allocate the new GB_jit_C_flags
    //--------------------------------------------------------------------------

    GB_COPY_STUFF (GB_jit_C_flags, new_C_flags) ;

    //--------------------------------------------------------------------------
    // allocate and define strings that depend on GB_jit_C_flags
    //--------------------------------------------------------------------------

    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_link_flags: return the current C link flags
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_link_flags (void)
{
    const char *s ;
    #pragma omp critical (GB_jitifyer_worker)
    {
        s = GB_jit_C_link_flags ;
    }
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_link_flags: set new C link flags
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_link_flags (const char *new_C_link_flags)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_C_link_flags == NULL)
    {
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C link flags in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    #pragma omp critical (GB_jitifyer_worker)
    { 
        info = GB_jitifyer_set_C_link_flags_worker (new_C_link_flags) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_link_flags_worker: set C link flags in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_link_flags_worker (const char *new_C_link_flags)
{

    //--------------------------------------------------------------------------
    // free the old strings that depend on the C flags
    //--------------------------------------------------------------------------

    GB_FREE_STUFF (GB_jit_C_link_flags) ;
    GB_FREE_STUFF (GB_jit_command) ;

    //--------------------------------------------------------------------------
    // allocate the new GB_jit_C_link_flags
    //--------------------------------------------------------------------------

    GB_COPY_STUFF (GB_jit_C_link_flags, new_C_link_flags) ;

    //--------------------------------------------------------------------------
    // allocate and define strings that depend on GB_jit_C_link_flags
    //--------------------------------------------------------------------------

    return (GB_jitifyer_alloc_space ( )) ;
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

    GrB_Info info ;
    if (hash == UINT64_MAX)
    { 
        // The kernel may not be compiled; it does not have a valid definition.
        GBURBLE ("(jit undefined) ") ;
        return (GrB_NO_VALUE) ;
    }

    if (GB_jit_control <= GxB_JIT_PAUSE)
    { 
        // The JIT control has disabled all JIT kernels.  Punt to generic.
        GBURBLE ("(jit paused) ") ;
        return (GrB_NO_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // handle the GxB_JIT_RUN case: critical section not required
    //--------------------------------------------------------------------------

    if (GB_jit_control == GxB_JIT_RUN)
    {

        //----------------------------------------------------------------------
        // look up the kernel in the hash table
        //----------------------------------------------------------------------

        (*dl_function) = GB_jitifyer_lookup (hash, encoding, suffix) ;
        if ((*dl_function) != NULL)
        { 
            // found the kernel in the hash table
            GBURBLE ("(jit run) ") ;
            return (GrB_SUCCESS) ;
        }
        else
        {
            // No kernels may be loaded or compiled, but existing kernels
            // already loaded may be run (handled above if dl_function was
            // found).  This kernel was not loaded, so punt to generic.
            GBURBLE ("(jit not loaded) ") ;
            return (GrB_NO_VALUE) ;
        }
    }

    //--------------------------------------------------------------------------
    // do the rest inside a critical section
    //--------------------------------------------------------------------------

    #pragma omp critical (GB_jitifyer_worker)
    {
        info = GB_jitifyer_worker (dl_function, family, kname, hash,
            encoding, suffix, semiring, monoid, op, type1, type2, type3) ;
    }

    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_worker: do the work for GB_jitifyer_load in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_worker
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
    // look up the kernel in the hash table
    //--------------------------------------------------------------------------

    (*dl_function) = GB_jitifyer_lookup (hash, encoding, suffix) ;
    if ((*dl_function) != NULL)
    { 
        // found the kernel in the hash table
        GBURBLE ("(jit run) ") ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // quick return if not in the hash table
    //--------------------------------------------------------------------------

    if (GB_jit_control <= GxB_JIT_RUN)
    { 
        // No kernels may be loaded or compiled, but existing kernels already
        // loaded may be run (handled above if dl_function was found).  This
        // kernel was not loaded, so punt to generic.
        GBURBLE ("(jit not loaded) ") ;
        return (GrB_NO_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // the kernel needs to be loaded, and perhaps compiled; get its properties
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
        "%s/lib%s%s", GB_jit_cache_path, kernel_name, GB_LIB_SUFFIX) ;
    // FIXME: dlopen only exists on Linux/Unix/Mac
    void *dl_handle = dlopen (GB_jit_library_name, RTLD_LAZY) ;

    //--------------------------------------------------------------------------
    // check if the kernel was found, but needs to be compiled anyway
    //--------------------------------------------------------------------------

    bool builtin = (encoding->suffix_len == 0) ;
    if (dl_handle != NULL && !builtin)
    { 
        // library is loaded but make sure the defn are OK
        // FIXME: dlsym only exists on Linux/Unix/Mac
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
            // FIXME: dlclose only exists on Linux/Unix/Mac
            dlclose (dl_handle) ;
            dl_handle = NULL ;
            if (GB_jit_control == GxB_JIT_LOAD)
            { 
                // If the JIT control is set to GxB_JIT_LOAD, new kernels
                // cannot be compiled.  This kernel has just been loaded but it
                // has stale definition.  Loading it again will result in the
                // same issue, but will take a lot of time if the kernel is
                // loaded again and again, since no new kernels can be
                // compiled.  Set the JIT control to GxB_JIT_RUN to avoid this
                // performance issue.
                GB_jit_control = GxB_JIT_RUN ;
                return (GrB_INVALID_VALUE) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // create and compile source file, if needed
    //--------------------------------------------------------------------------

    if (dl_handle == NULL)
    { 

        //----------------------------------------------------------------------
        // quick return if the JIT is not permitted to compile new kernels
        //----------------------------------------------------------------------

        if (GB_jit_control < GxB_JIT_ON)
        { 
            // No new kernels may be compiled, so punt to generic.
            GBURBLE ("(jit not compiled) ") ;
            return (GrB_NO_VALUE) ;
        }

        //----------------------------------------------------------------------
        // create the kernel source file
        //----------------------------------------------------------------------

        GBURBLE ("(jit compile and load) ") ;
        snprintf (GB_jit_kernel_name, GB_jit_kernel_name_allocated,
            "%s/%s.c", GB_jit_cache_path, kernel_name) ;
        FILE *fp = fopen (GB_jit_kernel_name, "w") ;
        if (fp == NULL)
        { 
            // disable the JIT to avoid repeated compilation errors
            GBURBLE ("(jit: cannot create kernel; compilation disabled) ") ;
            GB_jit_control = GxB_JIT_LOAD ;
            return (GrB_INVALID_VALUE) ;
        }
        fprintf (fp,
            "//--------------------------------------"
            "----------------------------------------\n"
            "// %s.c\n", kernel_name) ;
        GB_macrofy_copyright (fp) ;
        fprintf (fp, "#include \"GB_jit_kernel_%s.h\"\n\n", family_name) ;
        GB_macrofy_family (fp, family, encoding->code, semiring, monoid,
            op, type1, type2, type3) ;

        fprintf (fp, "#ifndef GB_JIT_RUNTIME\n"
                     "#define GB_jit_kernel %s\n"
                     "#endif\n"
                     "#include \"GB_jit_kernel_%s.c\"\n\n",
                     kernel_name, kname) ;
        fprintf (fp, "\n#ifdef GB_JIT_RUNTIME\n") ;
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
        fprintf (fp, "#endif\n") ;
        fclose (fp) ;

//      printf ("compile: %s\n", GB_jit_kernel_name) ;

        //----------------------------------------------------------------------
        // compile the source file to create the lib*.so file
        //----------------------------------------------------------------------

        GB_jitifyer_compile (kernel_name) ;
        // FIXME: dlopen only exists on Linux/Unix/Mac
        dl_handle = dlopen (GB_jit_library_name, RTLD_LAZY) ;
        if (dl_handle == NULL)
        { 
            // unable to open lib*.so file
//          printf ("cannot open lib: [%s]\n", GB_jit_library_name) ;
            GBURBLE ("(jit: compiler error; compilation disabled) ") ;
            // disable the JIT to avoid repeated compilation errors
            GB_jit_control = GxB_JIT_LOAD ;
            return (GrB_INVALID_VALUE) ;
        }
    }
    else
    { 
        GBURBLE ("(jit load) ") ;
    }

    //--------------------------------------------------------------------------
    // get the jit_kernel_function pointer
    //--------------------------------------------------------------------------

    // FIXME: dlsym only exists on Linux/Unix/Mac
    (*dl_function) = dlsym (dl_handle, "GB_jit_kernel") ;
    if ((*dl_function) == NULL)
    { 
        // unable to find GB_jit_kernel: punt to generic
        GBURBLE ("(jit: load error; JIT loading disabled) ") ;
        // FIXME: dlclose only exists on Linux/Unix/Mac
        dlclose (dl_handle) ; 
        // disable the JIT to avoid repeated loading errors
        GB_jit_control = GxB_JIT_RUN ;
        return (GrB_INVALID_VALUE) ;
    }

    // insert the new kernel into the hash table
    if (!GB_jitifyer_insert (hash, encoding, suffix, dl_handle, (*dl_function)))
    { 
        // unable to add kernel to hash table: punt to generic
        // FIXME: dlclose only exists on Linux/Unix/Mac
        dlclose (dl_handle) ; 
        // disable the JIT to avoid repeated errors
        GB_jit_control = GxB_JIT_PAUSE ;
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
        GB_FREE_STUFF (GB_jit_table) ;

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
// GB_jitifyer_table_free:  free the hash and clear all loaded kernels
//------------------------------------------------------------------------------

// After calling this function, the JIT is still enabled.  GB_jitifyer_insert
// will reallocate the table if it is NULL.

void GB_jitifyer_table_free (void)
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
                // FIXME: dlclose only exists on Linux/Unix/Mac
                dlclose (e->dl_handle) ;
            }
        }
    }

    GB_FREE_STUFF (GB_jit_table) ;
    GB_jit_table_size = 0 ;
    GB_jit_table_bits = 0 ;
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
    // FIXME: dlsym only exists on Linux/Unix/Mac
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
    // FIXME: dlsym only exists on Linux/Unix/Mac
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

    // compile:
    "%s -DGB_JIT_RUNTIME "          // compiler command
    #ifdef GBRENAME
    "-DGBRENAME=1 "                 // rename for MATLAB
    #endif
    "%s "                           // C flags
    "%s "                           // include directories
    "%s "                           // openmp include directories
    "-o %s/%s%s "                   // *.o output file
    "-c %s/%s.c ; "                 // *.c input file

    // link:
    "%s "                           // C compiler
    "%s "                           // C flags
    "%s "                           // C link flags
    "-o %s/lib%s%s "                // lib*.so output file
    "%s/%s%s "                      // *.o input file
    // FIXME: add libgraphblas to GB_LIBRARIES
    // allow GB_LIBRARIES to be modified
    "%s%s/build/libgraphblas%s%s"   // libgraphblas.so
    "%s "                           // libraries to link with
    ,

    // compile:
    GB_jit_C_compiler,                              // C compiler
    GB_jit_C_flags,                                 // C flags
    GB_jit_include,                                 // include directories
    GB_OMP_INC,                                     // openmp include
    GB_jit_cache_path, kernel_name, GB_OBJ_SUFFIX,  // *.o output file
    GB_jit_cache_path, kernel_name,                 // *.c input file

    // link:
    GB_jit_C_compiler,                              // C compiler
    GB_jit_C_flags,                                 // C flags
    GB_jit_C_link_flags,                            // C link flags
    GB_jit_cache_path, kernel_name, GB_LIB_SUFFIX,  // lib*.so output file
    GB_jit_cache_path, kernel_name, GB_OBJ_SUFFIX,  // *.o input file
    GB_jit_source_path,                             // libgraphblas.so
    #ifdef GBRENAME
    "/GraphBLAS", "_matlab"
    #else
    "", "",
    #endif
    GB_LIB_SUFFIX,
    GB_LIBRARIES) ;                 // libraries to link with

    GBURBLE ("(jit compile: %s) ", GB_jit_command) ;

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

