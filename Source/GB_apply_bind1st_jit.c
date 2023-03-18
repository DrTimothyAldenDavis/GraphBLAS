//------------------------------------------------------------------------------
// GB_apply_bind1st_jit: Cx=op(x,B) apply bind1st method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_apply.h"
#include "GB_ewise_kernels.h"
#include "GB_stringify.h"
#include "GB_jitifyer.h"

typedef GrB_Info (*GB_jit_dl_function)
(
    GB_void *Cx,
    const GB_void *xscalar,
    const GB_void *Bx,
    const int8_t *restrict Bb,
    const int64_t bnz,
    const int nthreads
) ;

GrB_Info GB_apply_bind1st_jit   // Cx = op (x,B), apply bind1st via the JIT
(
    const char *kname,          // kernel name
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GrB_BinaryOp binaryop,
    const GB_void *xscalar,
    const GrB_Matrix B,
    const int nthreads
)
{

#ifdef GBRENAME
    return (GrB_NO_VALUE) ;
#else

    //------------------------------------------------------------------
    // enumify the problem and look it up in the jit hash
    //------------------------------------------------------------------

    GrB_Info info ;
    GBURBLE ("(jit) ") ;
    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_KERNEL_APPLYBIND1, false,
        false, false, GxB_FULL, ctype, NULL, false, false,
        binaryop, false, NULL, B) ;
    if (hash == UINT64_MAX)
    {
        // cannot JIT this binaryop
        return (GrB_NO_VALUE) ;
    }
    void *dl_function = GB_jitifyer_lookup (hash, &encoding, suffix) ;

    //------------------------------------------------------------------
    // load it and compile it if not found
    //------------------------------------------------------------------

    if (dl_function == NULL)
    { 

        //----------------------------------------------------------------------
        // name and load the jit kernel
        //----------------------------------------------------------------------

        char kernel_name [GB_KLEN] ;
        char lib_filename [2048] ;
        void *dl_handle ;
        FILE *fp ;
        info = GB_jitifyer_load (&dl_handle, &fp, kernel_name, lib_filename,
            kname, 13, &encoding, suffix,
            NULL, (GB_Operator) binaryop, ctype, NULL, B->type) ;
        if (info != GrB_SUCCESS) return (info) ;

        //--------------------------------------------------------------
        // compile the jit kernel, if not found or if op/type changed
        //--------------------------------------------------------------

        if (dl_handle == NULL)
        {

            fprintf (fp, "\n#include \"GB_jit_kernel_ewise.h\"\n") ;
            GB_macrofy_ewise (fp, encoding.code, binaryop, ctype, NULL,
                B->type) ;
            fprintf (fp, "\n#include \"GB_jit_kernel_%s.c\"\n", kname) ;

            bool builtin = (encoding.suffix_len == 0) ;
            if (!builtin)
            {
                // create query_defn function
                GB_macrofy_query_defn (fp,
                    NULL,
                    (GB_Operator) binaryop,
                    ctype, NULL, B->type) ;
            }

            GB_macrofy_query_version (fp) ;
            fclose (fp) ;

            //----------------------------------------------------------
            // compile the *.c file to create the lib*.so file
            //----------------------------------------------------------

            GB_jitifyer_compile (kernel_name) ;
            dl_handle = dlopen (lib_filename, RTLD_LAZY) ;
            if (dl_handle == NULL)
            {
                // unable to open lib*.so file: punt to generic
                printf ("failed to load lib*.so file!\n") ;
                return (GrB_PANIC) ;
            }
        }
        else
        {
            GBURBLE ("(loaded) ") ;
        }

        // get the jit_kernel_function pointer
        dl_function = dlsym (dl_handle, "GB_jit_kernel") ;
        if (dl_function == NULL)
        {
            // unable to find GB_jit_kernel: punt to generic
            printf ("failed to load dl_function!\n") ;
            dlclose (dl_handle) ; 
            return (GrB_PANIC) ;
        }

        // insert the new kernel into the hash table
        if (!GB_jitifyer_insert (hash, &encoding, suffix,
            dl_handle, dl_function))
        {
            // unable to add kernel to hash table: punt to generic
            dlclose (dl_handle) ; 
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //------------------------------------------------------------------
    // call the jit kernel and return result
    //------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    info = GB_jit_kernel (Cx, xscalar, B->x, B->b,
        GB_nnz_held (B), nthreads) ;
    return (info) ;
#endif
}

