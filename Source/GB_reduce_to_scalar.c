//------------------------------------------------------------------------------
// GB_reduce_to_scalar: reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// c = accum (c, reduce_to_scalar(A)), reduce entries in a matrix to a scalar.
// Does the work for GrB_*_reduce_TYPE, both matrix and vector.

// This function does not need to know if A is hypersparse or not, and its
// result is the same if A is in CSR or CSC format.

// This function is the only place in all of GraphBLAS where the identity value
// of a monoid is required, but only in one special case: it is required to be
// the return value of c when A has no entries.  The identity value is also
// used internally, in the parallel methods below, to initialize a scalar value
// in each task.  The methods could be rewritten to avoid the use of the
// identity value.  Since this function requires it anyway, for the special
// case when nvals(A) is zero, the existence of the identity value makes the
// code a little simpler.

#include "GB_reduce.h"
#include "GB_binop.h"
#include "GB_stringify.h"
#ifndef GBCUDA_DEV
#include "GB_red__include.h"
#endif

#include "GB_jitifyer.h"
#include <dlfcn.h>

typedef GrB_Info (*GB_reduce_function)
    (void *result, const GrB_Matrix, void *, bool *, int, int) ;

#define GB_FREE_ALL                 \
{                                   \
    GB_WERK_POP (F, bool) ;         \
    GB_WERK_POP (W, GB_void) ;      \
}

GrB_Info GB_reduce_to_scalar    // z = reduce_to_scalar (A)
(
    void *c,                    // result scalar
    const GrB_Type ctype,       // the type of scalar, c
    const GrB_BinaryOp accum,   // for c = accum(c,z)
    const GrB_Monoid monoid,    // monoid to do the reduction
    const GrB_Matrix A,         // matrix to reduce
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL (c) ;
    GB_WERK_DECLARE (W, GB_void) ;
    GB_WERK_DECLARE (F, bool) ;

    ASSERT_TYPE_OK (ctype, "type of scalar c", GB0) ;
    ASSERT_MONOID_OK (monoid, "monoid for reduce_to_scalar", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for reduce_to_scalar", GB0) ;
    ASSERT_MATRIX_OK (A, "A for reduce_to_scalar", GB0) ;

    // check domains and dimensions for c = accum (c,z)
    GrB_Type ztype = monoid->op->ztype ;
    GB_OK (GB_compatible (ctype, NULL, NULL, false, accum, ztype, Werk)) ;

    // z = monoid (z,A) must be compatible
    if (!GB_Type_compatible (A->type, ztype))
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // assemble any pending tuples; zombies are OK
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_PENDING (A) ;
    GB_BURBLE_DENSE (A, "(A %s) ") ;

    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t asize = A->type->size ;
    int64_t zsize = ztype->size ;
    int64_t anz = GB_nnz_held (A) ;
    ASSERT (anz >= A->nzombies) ;

    // z = identity
    GB_void z [GB_VLA(zsize)] ;
    memcpy (z, monoid->identity, zsize) ;   // required, if nnz(A) is zero

    #ifdef GB_DEBUGIFY_DEFN
    // FIXME: this will move below
    GB_debugify_reduce (monoid, A) ;
    #endif

    //--------------------------------------------------------------------------
    // z = reduce_to_scalar (A) on the GPU(s) or CPU
    //--------------------------------------------------------------------------

    #if defined ( GBCUDA )
    if (GB_reduce_to_scalar_cuda_branch (monoid, A))
    {

        //----------------------------------------------------------------------
        // use the GPU(s)
        //----------------------------------------------------------------------

        GB_OK (GB_reduce_to_scalar_cuda (z, monoid, A)) ;

    }
    else
    #endif
    {

        //----------------------------------------------------------------------
        // use OpenMP on the CPU threads
        //----------------------------------------------------------------------

        bool done = false ;
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;
        int ntasks = (nthreads == 1) ? 1 : (64 * nthreads) ;
        ntasks = GB_IMIN (ntasks, anz) ;
        ntasks = GB_IMAX (ntasks, 1) ;

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        GB_WERK_PUSH (W, ntasks * zsize, GB_void) ;
        GB_WERK_PUSH (F, ntasks, bool) ;
        if (W == NULL || F == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // z = reduce_to_scalar (A)
        //----------------------------------------------------------------------

        // get terminal value, if any
        GB_void *restrict zterminal = (GB_void *) monoid->terminal ;

        if (anz == A->nzombies)
        { 

            //------------------------------------------------------------------
            // no live entries in A; nothing to do
            //------------------------------------------------------------------

            done = true ;

        }
        else if (A->iso)
        { 

            //------------------------------------------------------------------
            // reduce an iso matrix to scalar
            //------------------------------------------------------------------

            // this takes at most O(log(nvals(A))) time, for any monoid
            GB_iso_reduce_to_scalar (z, monoid, A) ;
            done = true ;

        }
        else if (A->type == ztype)
        {

            //------------------------------------------------------------------
            // reduce to scalar via built-in operator
            //------------------------------------------------------------------

            #ifndef GBCUDA_DEV

                //--------------------------------------------------------------
                // define the worker for the switch factory
                //--------------------------------------------------------------

                #define GB_red(opname,aname) \
                    GB (_red_ ## opname ## aname)

                #define GB_RED_WORKER(opname,aname,ztype)                   \
                {                                                           \
                    info = GB_red (opname, aname) ((ztype *) z, A, W, F,    \
                        ntasks, nthreads) ;                                 \
                    done = (info != GrB_NO_VALUE) ;                         \
                }                                                           \
                break ;

                //--------------------------------------------------------------
                // launch the switch factory
                //--------------------------------------------------------------

                // controlled by opcode and typecode
                GB_Opcode opcode = monoid->op->opcode ;
                GB_Type_code typecode = A->type->code ;
                ASSERT (typecode <= GB_UDT_code) ;

                #include "GB_red_factory.c"

            #endif
        }

        //----------------------------------------------------------------------
        // use JIT worker
        //----------------------------------------------------------------------

        #ifdef GB_DEBUGIFY_DEFN
        if (!done)
        {

            //------------------------------------------------------------------
            // enumify the reduce problem, including all objects and types
            //------------------------------------------------------------------

            GB_jit_encoding encoding ;
            char suffix [8 + 2*GxB_MAX_NAME_LEN] ;
            uint64_t hash = GB_encodify_reduce (&encoding, suffix, monoid, A) ;
//          printf ("hash: %016" PRIx64 "\n", hash) ;
//          printf ("suffix: [%s]\n", suffix) ;

            //------------------------------------------------------------------
            // find the kernel in the global hash table
            //------------------------------------------------------------------

            void *dl_function = GB_jitifyer_lookup (hash, &encoding, suffix) ;

            //------------------------------------------------------------------
            // load it and compile it if not found
            //------------------------------------------------------------------

            if (dl_function == NULL)
            { 

                //--------------------------------------------------------------
                // first time this kernel has been seen since GrB_init
                //--------------------------------------------------------------

                // namify the reduce problem
                #define RLEN (256 + 2 * GxB_MAX_NAME_LEN)
                char reduce_name [RLEN] ;
                uint64_t rcode = encoding.code ;

                snprintf (reduce_name, RLEN-1, "GB_jit_reduce_%0*" PRIx64 "%s",
                    7, rcode, suffix) ;
//              printf ("name: [%s]\n", reduce_name) ;

                /*
                bool builtin = encoding.primary.builtin ;
                GB_namify_problem (reduce_name, "GB_jit_reduce_", 7, rcode,
                    builtin,
                    monoid->op->name,
                    NULL,
                    monoid->op->ztype->name,
                    A->type->name,
                    NULL,
                    NULL,
                    NULL,
                    NULL) ;
                */

                //==============================================================
                // FIXME: make this a helper function for all kernels

                // FIXME: create this at GrB_init time, or by GxB_set
                char lib_folder [2048] ;
                snprintf (lib_folder, 2047,
                    "/home/faculty/d/davis/.SuiteSparse/GraphBLAS/v%d.%d.%d",
                    GxB_IMPLEMENTATION_MAJOR,
                    GxB_IMPLEMENTATION_MINOR,
                    GxB_IMPLEMENTATION_SUB) ;

                // try to load the libkernelname.so from the user's
                // .SuiteSparse/GraphBLAS folder (if already compiled)
                char lib_filename [2048] ;
                snprintf (lib_filename, 2048, "%s/lib%s.so",
                    lib_folder, reduce_name) ;
                void *dl_handle = dlopen (lib_filename, RTLD_LAZY) ;

                bool need_to_compile = (dl_handle == NULL) ;
                bool builtin = (encoding.suffix_len == 0) ;

                if (!need_to_compile && !builtin)
                {
                    // already compiled, but make sure the defn are OK
                    #define QLEN RLEN+32
                    char query_name [QLEN] ;
                    snprintf (query_name, QLEN, "%s__query_defn", reduce_name) ;
                    void *dl_query = dlsym (dl_handle, query_name) ;
                    if (dl_query == NULL)
                    {
                        // the library is invalid; recompile it
                        need_to_compile = true ;
                    }
                    else
                    {
                        GB_jit_query_defn_function query_defn = (GB_jit_query_defn_function) dl_query ;

                        // FIXME: make this a GB_jitifyer_helper function
                        // compare the monoid definition
                        const char *opdef = query_defn (0) ;
                        if (opdef != NULL) printf ("opdef: \n%s\n", opdef) ;
                        if ((monoid->op->defn != NULL) != (opdef != NULL))
                        {
                            // one is not NULL but the other is NULL
                            need_to_compile = true ;
                        }
                        else if (monoid->op->defn != NULL)
                        {
                            // ensure the user-defined monoid hasn't changed
                            printf ("monoid def: \n%s\n", monoid->op->defn) ;
                            need_to_compile = (strcmp (opdef, monoid->op->defn) != 0) ;
                        }

                        // FIXME: make this a GB_jitifyer_helper function
                        // compare the type definition
                        if (!need_to_compile)
                        {
                            const char *tdef = query_defn (2) ;
                            if (tdef != NULL) printf ("tdef: \n%s\n", tdef) ;
                            if ((A->type->defn != NULL) != (tdef != NULL))
                            {
                                // one is not NULL but the other is NULL
                                need_to_compile = true ;
                            }
                            else if (A->type->defn != NULL)
                            {
                                // ensure the user-defined type hasn't changed
                                printf ("A type def: \n%s\n", A->type->defn) ;
                                need_to_compile = (strcmp (tdef, A->type->defn) != 0) ;
                            }
                        }

                        // FIXME: make this a GB_jitifyer_helper function
                        // compare the identity and terminal
                        if (!need_to_compile)
                        {
                            snprintf (query_name, QLEN, "%s__query_monoid", reduce_name) ;
                            dl_query = dlsym (dl_handle, query_name) ;
                            if (dl_query == NULL)
                            {
                                // the library is invalid; recompile it
                                need_to_compile = true ;
                            }
                            else
                            {
                                // check the identity and terminal values
                                GB_jit_query_monoid_function query_monoid = (GB_jit_query_monoid_function) dl_query ;
                                size_t tsize = (monoid->terminal == NULL) ? 0 : zsize ;
                                if (!query_monoid (monoid->identity, monoid->terminal, zsize, tsize))
                                {
                                    // wrong sizes or value(s); need to recompile
                                    need_to_compile = true ;
                                }
                            }
                        }
                    }

                    if (need_to_compile)
                    {
                        // library is loaded but needs to change, so close it
                        dlclose (dl_handle) ;
                    }
                }

                if (need_to_compile)
                {

                    //----------------------------------------------------------
                    // construct a new jit kernel for this instance
                    //----------------------------------------------------------

                    printf ("compiling the kernel\n") ;
                    char source_filename [2048] ;
                    snprintf (source_filename, 2048, "%s/%s.c",
                        lib_folder, reduce_name) ;
//                  printf ("source file: %s\n", source_filename) ;
                    FILE *fp = fopen (source_filename, "w") ;
                    if (fp == NULL)
                    {
                        // FIXME
                        printf ("failed to write to *.c file!\n") ;
                        fflush (stdout) ;
                        abort ( ) ;
                        return (GrB_PANIC) ;
                    }
                    fprintf (fp,
                        "//--------------------------------------"
                        "----------------------------------------\n") ;
                    fprintf (fp, "// %s.h\n"
                        "#include \"GB_jit_kernel_reduce.h\"\n"
                        "#define GB_JIT_KERNEL %s\n",
                        reduce_name, reduce_name) ;
                    GB_macrofy_reduce (fp, rcode, monoid, A->type) ;
                    fprintf (fp,
                        "\n// reduction kernel\n"
                        "#include \"GB_jit_kernel_reduce.c\"\n") ;

                    if (!builtin)
                    {
                        // either the monoid or A->type is not builtin, or both
                        GB_macrofy_query_defn (fp, reduce_name,
                            (GB_Operator) monoid->op, NULL,
                            A->type, NULL, NULL, NULL, NULL, NULL) ;
                    }
                    if (!monoid->builtin)
                    {
                        // the monoid is not builtin
                        GB_macrofy_query_monoid (fp, reduce_name, monoid) ;
                    }
                    fclose (fp) ;

                    //----------------------------------------------------------
                    // compile the *.c file to create the lib*.so file
                    //----------------------------------------------------------

                    // FIXME: create this at GrB_init time
                    char root_folder [256] ;
                    snprintf (root_folder, 256, "%s",
                    "/home/faculty/d/davis/cuda/GraphBLAS") ;

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
                        "-I%s/cpu_features/include",
                        lib_folder,
                        root_folder,
                        root_folder,
                        root_folder,
                        root_folder,
                        root_folder,
                        root_folder,
                        root_folder,
                        root_folder) ;

                    char command [4096] ;

                    snprintf (command, 4096,
                    "gcc -fPIC -O3 -std=c11 -fexcess-precision=fast "
                    "-fcx-limited-range -fno-math-errno -fwrapv -DNDEBUG "
                    "-fopenmp %s -o"
                    "%s/%s.o -c %s/%s.c ;" 
                    "gcc -fPIC -O3 -std=c11 -fexcess-precision=fast "
                    "-fcx-limited-range -fno-math-errno -fwrapv -DNDEBUG "
                    "-fopenmp "
                    " -shared -Wl,-soname,lib%s.so -o %s/lib%s.so"
                    " %s/%s.o -lm",
                    include_files,
                    lib_folder, reduce_name,    // *.o file, first gcc command
                    lib_folder, reduce_name,    // *.c file, first gcc command
                    reduce_name,                // soname
                    lib_folder, reduce_name,    // lib*.so output file
                    lib_folder, reduce_name) ;  // *.o file for 2nd gcc

//                  printf ("%s\n", command) ;

                    int result = system (command) ;
//                  printf ("result: %d\n", result) ;
//                  printf ("lib_file: %s\n", lib_filename) ;
                
                    // load in the lib*.so file
                    dl_handle = dlopen (lib_filename, RTLD_LAZY) ;
                    if (dl_handle == NULL)
                    {
                        // FIXME
                        printf ("failed to load lib*.so file!\n") ;
                        fflush (stdout) ;
                        abort ( ) ;
                        return (GrB_PANIC) ;
                    }
                }

                // get the jit_kernel_function pointer
                dl_function = dlsym (dl_handle, reduce_name) ;
                if (dl_function == NULL)
                {
                    // FIXME
                    printf ("failed to load dl_function!\n") ;
                    fflush (stdout) ;
                    abort ( ) ;
                    return (GrB_PANIC) ;
                }

                // insert the new kernel into the hash table
                if (!GB_jitifyer_insert (hash, &encoding, suffix, dl_handle,
                    dl_function))
                {
                    // FIXME
                    printf ("failed to add to hash table!\n") ;
                    fflush (stdout) ;
                    abort ( ) ;
                    return (GrB_PANIC) ;
                }

                //==============================================================
            }
            else
            {
//              printf ("found in hash: %p\n", dl_function) ;
            }

            // call the kernel
            GB_reduce_function redfunc = (GB_reduce_function) dl_function ;
            info = redfunc (z, A, W, F, ntasks, nthreads) ;

            // set true if JIT successful
            done = (info == GrB_SUCCESS) ;

        }
        #endif

        //----------------------------------------------------------------------
        // use generic worker
        //----------------------------------------------------------------------

        if (!done)
        {

            //------------------------------------------------------------------
            // generic worker
            //------------------------------------------------------------------

            if (A->type == ztype)
            { 

                //--------------------------------------------------------------
                // generic worker: sum up the entries, no typecasting
                //--------------------------------------------------------------

                GB_BURBLE_MATRIX (A, "(generic reduce to scalar: %s) ",
                    monoid->op->name) ;

                // the switch factory didn't handle this case
                GxB_binary_function freduce = monoid->op->binop_function ;

                #define GB_A_TYPENAME GB_void

                // no panel used
                #define GB_PANEL 1
                #define GB_NO_PANEL_CASE

                // ztype z = identity
                #define GB_DECLARE_MONOID_IDENTITY(z)                   \
                    GB_void z [GB_VLA(zsize)] ;                         \
                    memcpy (z, monoid->identity, zsize) ;

                // W [tid] = t, no typecast
                #define GB_COPY_SCALAR_TO_ARRAY(W, tid, t)              \
                    memcpy (W +(tid*zsize), t, zsize)

                // z += W [k], no typecast
                #define GB_ADD_ARRAY_TO_SCALAR(z,W,k)                   \
                    freduce (z, z, W +((k)*zsize))

                // t += (ztype) Ax [p], but no typecasting needed
                #define GB_GETA_AND_UPDATE(t,Ax,p)                      \
                    freduce (t, t, Ax +((p)*zsize))

                // terminal condition (not used if the monoid is not terminal)
                #define GB_TERMINAL_CONDITION(z,zterminal)              \
                    (memcmp (z, zterminal, zsize) == 0)
                #define GB_IF_TERMINAL_BREAK(z,zterminal)               \
                    if (GB_TERMINAL_CONDITION (z, zterminal)) break ;

                if (zterminal == NULL)
                {
                    // monoid is not terminal
                    #define GB_MONOID_IS_TERMINAL 0
                    #include "GB_reduce_to_scalar_template.c"
                }
                else
                {
                    // break if terminal value reached
                    #undef  GB_MONOID_IS_TERMINAL
                    #define GB_MONOID_IS_TERMINAL 1
                    #include "GB_reduce_to_scalar_template.c"
                }

            }
            else
            { 

                //--------------------------------------------------------------
                // generic worker: sum up the entries, with typecasting
                //--------------------------------------------------------------

                GB_BURBLE_MATRIX (A, "(generic reduce to scalar, with typecast:"
                    " %s) ", monoid->op->name) ;

                GxB_binary_function freduce = monoid->op->binop_function ;
                GB_cast_function
                    cast_A_to_Z = GB_cast_factory (ztype->code, A->type->code) ;

                // t += (ztype) Ax [p], with typecast
                #undef  GB_GETA_AND_UPDATE
                #define GB_GETA_AND_UPDATE(t,Ax,p)                      \
                    GB_void awork [GB_VLA(zsize)] ;                     \
                    cast_A_to_Z (awork, Ax +((p)*asize), asize) ;       \
                    freduce (t, t, awork)

                if (zterminal == NULL)
                {
                    // monoid is not terminal
                    #undef  GB_MONOID_IS_TERMINAL
                    #define GB_MONOID_IS_TERMINAL 0
                    #include "GB_reduce_to_scalar_template.c"
                }
                else
                {
                    // break if terminal value reached
                    #undef  GB_MONOID_IS_TERMINAL
                    #define GB_MONOID_IS_TERMINAL 1
                    #include "GB_reduce_to_scalar_template.c"
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // c = z or c = accum (c,z)
    //--------------------------------------------------------------------------

    // This operation does not use GB_accum_mask, since c and z are
    // scalars, not matrices.  There is no scalar mask.

    if (accum == NULL)
    { 
        // c = (ctype) z
        GB_cast_function
            cast_Z_to_C = GB_cast_factory (ctype->code, ztype->code) ;
        cast_Z_to_C (c, z, ctype->size) ;
    }
    else
    { 
        GxB_binary_function faccum = accum->binop_function ;

        GB_cast_function cast_C_to_xaccum, cast_Z_to_yaccum, cast_zaccum_to_C ;
        cast_C_to_xaccum = GB_cast_factory (accum->xtype->code, ctype->code) ;
        cast_Z_to_yaccum = GB_cast_factory (accum->ytype->code, ztype->code) ;
        cast_zaccum_to_C = GB_cast_factory (ctype->code, accum->ztype->code) ;

        // scalar workspace
        GB_void xaccum [GB_VLA(accum->xtype->size)] ;
        GB_void yaccum [GB_VLA(accum->ytype->size)] ;
        GB_void zaccum [GB_VLA(accum->ztype->size)] ;

        // xaccum = (accum->xtype) c
        cast_C_to_xaccum (xaccum, c, ctype->size) ;

        // yaccum = (accum->ytype) z
        cast_Z_to_yaccum (yaccum, z, zsize) ;

        // zaccum = xaccum "+" yaccum
        faccum (zaccum, xaccum, yaccum) ;

        // c = (ctype) zaccum
        cast_zaccum_to_C (c, zaccum, ctype->size) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

