//------------------------------------------------------------------------------
// GB_subassign_jit: interface to JIT kernels for all assign/subassign methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// FIXME: debug is on
#define GB_DEBUG

#include "GB.h"
#include "GB_stringify.h"
#include "GB_jitifyer.h"

typedef GrB_Info (*GB_jit_dl_function)
(
    // input/output:
    GrB_Matrix C,
    // input:
    // I:
    const GrB_Index *I,
    const int64_t ni,
    const int64_t nI,
    const int64_t Icolon [3],
    // J:
    const GrB_Index *J,
    const int64_t nj,
    const int64_t nJ,
    const int64_t Jcolon [3],
    // mask M:
    const GrB_Matrix M,
    // A matrix or scalar:
    const GrB_Matrix A,
    const void *scalar,
    GB_Werk Werk
) ;

GrB_Info GB_subassign_jit
(
    // input/output:
    GrB_Matrix C,
    // input:
    const bool C_replace,
    // I:
    const GrB_Index *I,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    // J:
    const GrB_Index *J,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    // mask M:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    // accum, if present:
    const GrB_BinaryOp accum,   // may be NULL
    // A matrix or scalar:
    const GrB_Matrix A,         // NULL for scalar assignment
    const void *scalar,
    const GrB_Type scalar_type,
    // kind and kernel:
    const int assign_kind,      // row assign, col assign, assign, or subassign
    const char *kname,          // "sub01", etc
    const int assign_kernel,    // GB_JIT_KERNEL_SUBASSIGN_01, ... etc
    GB_Werk Werk
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
    uint64_t hash = GB_encodify_assign (&encoding, &suffix,
        assign_kernel, C, C_replace, Ikind, Jkind, M, Mask_struct,
        Mask_comp, accum, A, scalar_type, assign_kind) ;
    if (hash == UINT64_MAX)
    {
        // cannot JIT this accum or type
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
                "GB_jit_%s_%0*" PRIx64, kname, 12, scode) ;
        }
        else
        {
            snprintf (kernel_name, KLEN-1,
                "GB_jit_%s_%0*" PRIx64 "__%s", kname, 12, scode, suffix) ;
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

        GrB_Type atype = (A == NULL) ? scalar_type : A->type ;

        if (!need_to_compile && !builtin)
        {
            // not loaded but already compiled; make sure the defn are OK
            void *dl_query = dlsym (dl_handle, "GB_jit_query_defn") ;
            need_to_compile =
                !GB_jitifyer_match_version (dl_handle) ||
                !GB_jitifyer_match_defn (dl_query, 1, accum->defn) ||
                !GB_jitifyer_match_defn (dl_query, 2, C->type->defn) ||
                !GB_jitifyer_match_defn (dl_query, 3, atype->defn) ;
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
                "#include \"GB_jit_kernel_assign.h\"\n",
                kernel_name) ;
            // create query_version function
            GB_macrofy_query_version (fp) ;
            // }

            GB_macrofy_assign (fp, scode, accum, C->type, atype) ;
            fprintf (fp, "\n#include \"GB_jit_kernel_%s.c\"\n", kname) ;

            if (!builtin)
            {
                // create query_defn function
                GB_macrofy_query_defn (fp,
                    NULL,
                    (GB_Operator) accum,
                    C->type, atype, NULL) ;
            }

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
    GrB_Info info = GB_jit_kernel (C, I, ni, nI, Icolon,
        J, nj, nJ, Jcolon, M, A, scalar, Werk) ;
    return (info) ;
#endif
}

