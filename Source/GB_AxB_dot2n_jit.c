//------------------------------------------------------------------------------
// GB_AxB_dot2n_jit: C<M>=A'*B dot2n method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_stringify.h"
#include "GB_jitifyer.h"

typedef GrB_Info (*GB_jit_dl_function)
(
    GrB_Matrix,
    const GrB_Matrix,
    const GrB_Matrix, int64_t *restrict,
    const GrB_Matrix, int64_t *restrict,
    const int,
    const int,
    const int
) ;

GrB_Info GB_AxB_dot2n_jit        // C<M>=A'*B, dot2n method, via the JIT
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A, int64_t *restrict A_slice,
    const GrB_Matrix B, int64_t *restrict B_slice,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int nthreads,
    const int naslice,
    const int nbslice
)
{

#ifdef GBRENAME
    return (GrB_NO_VALUE) ;
#else

    //------------------------------------------------------------------
    // enumify the problem and look it up in the jit hash
    //------------------------------------------------------------------

    GBURBLE ("(jit) ") ;
    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_mxm (&encoding, &suffix,
        GB_JIT_KERNEL_MXM_DOT2N,
        C->iso, false, GB_sparsity (C), C->type,
        M, Mask_struct, Mask_comp, semiring, flipxy, A, B) ;
    if (hash == UINT64_MAX)
    {
        // cannot JIT this semiring
        return (GrB_NO_VALUE) ;
    }
    void *dl_function = GB_jitifyer_lookup (hash, &encoding, suffix) ;

    //------------------------------------------------------------------
    // load it and compile it if not found
    //------------------------------------------------------------------

    if (dl_function == NULL)
    { 

        //--------------------------------------------------------------
        // first time this kernel has been seen since GrB_init
        //--------------------------------------------------------------

        // namify the problem
        #define KLEN (256 + 2*GxB_MAX_NAME_LEN)
        char kernel_name [KLEN] ;
        uint64_t scode = encoding.code ;
        if (suffix == NULL)
        {
            snprintf (kernel_name, KLEN-1,
                "GB_jit_AxB_dot2n_%0*" PRIx64, 16, scode) ;
        }
        else
        {
            snprintf (kernel_name, KLEN-1,
                "GB_jit_AxB_dot2n_%0*" PRIx64 "__%s", 16, scode, suffix) ;
        }

        char lib_filename [2048] ;

        //==============================================================
        // FIXME: make this a helper function for all kernels
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
        // try to load the libkernelname.so from the user's
        // .SuiteSparse/GraphBLAS folder (if already compiled)
        snprintf (lib_filename, 2048, "%s/lib%s.so", lib_folder, kernel_name) ;

//      char command [4096] ;
//      sprintf (command, "ldd %s\n", lib_filename) ;
//      int res = system (command) ;
//      printf ("result: %d\n", res) ;
//      sprintf (command, "readelf -d %s\n", lib_filename) ;
//      res = system (command) ;
//      printf ("result: %d\n", res) ;

        void *dl_handle = dlopen (lib_filename, RTLD_LAZY) ;

        bool need_to_compile = (dl_handle == NULL) ;
        bool builtin = (encoding.suffix_len == 0) ;
        GrB_Monoid monoid = semiring->add ;

        if (!need_to_compile && !builtin)
        {
            // not loaded but already compiled; make sure the defn are OK
            void *dl_query = dlsym (dl_handle, "GB_jit_query_defn") ;
            need_to_compile =
                !GB_jitifyer_match_version (dl_handle) ||
                !GB_jitifyer_match_defn (dl_query, 0, monoid->op->defn) ||
                !GB_jitifyer_match_defn (dl_query, 1, semiring->multiply->defn) ||
                !GB_jitifyer_match_defn (dl_query, 2, C->type->defn) ||
                !GB_jitifyer_match_defn (dl_query, 3, A->type->defn) ||
                !GB_jitifyer_match_defn (dl_query, 4, B->type->defn) ||
                !GB_jitifyer_match_idterm (dl_handle, monoid) ;
            if (need_to_compile)
            {
                // library is loaded but needs to change, so close it
                dlclose (dl_handle) ;
            }
        }

        //--------------------------------------------------------------
        // compile the jit kernel, if not found or if op/type changed
        //--------------------------------------------------------------

        if (need_to_compile)
        {

            //----------------------------------------------------------
            // construct a new jit kernel for this instance
            //----------------------------------------------------------

            // {
            GBURBLE ("(compiling) ") ;
            char source_filename [2048] ;
            snprintf (source_filename, 2048, "%s/%s.c",
                lib_folder, kernel_name) ;
            FILE *fp = fopen (source_filename, "w") ;
            if (fp == NULL)
            {
                // unable to open source file: punt to generic
                printf ("failed to write to *.c file!\n") ;
                return (GrB_PANIC) ;
            }
            fprintf (fp,
                "//--------------------------------------"
                "----------------------------------------\n") ;
            fprintf (fp, "// %s.c\n"
                "#include \"GB_jit_kernel_AxB.h\"\n",
                kernel_name) ;
            // create query_version function
            GB_macrofy_query_version (fp) ;
            // }

            GB_macrofy_mxm (fp, scode, semiring, C->type, A->type, B->type) ;
            fprintf (fp, "\n#include \"GB_jit_kernel_AxB_dot2n.c\"\n") ;

            if (!builtin)
            {
                // create query_defn function
                GB_macrofy_query_defn (fp,
                    (GB_Operator) monoid->op,
                    (GB_Operator) semiring->multiply,
                    C->type, A->type, B->type) ;
            }

            // create query_monoid function if the monoid is not builtin
            GB_macrofy_query_monoid (fp, monoid) ;
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
            printf ("punt to generic\n") ;
            dlclose (dl_handle) ; 
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //------------------------------------------------------------------
    // call the jit kernel and return result
    //------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    GrB_Info info = GB_jit_kernel (C, M, A, A_slice, B, B_slice,
        nthreads, naslice, nbslice) ;
    return (info) ;
#endif
}

