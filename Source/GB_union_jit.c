//------------------------------------------------------------------------------
// GB_union_jit: C=A+B, C<#M>=A+B eWiseUnion method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_add.h"
#include "GB_ewise_kernels.h"
#include "GB_stringify.h"
#include "GB_jitifyer.h"

typedef GrB_Info (*GB_jit_dl_function)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_void *alpha_scalar_in,
    const GB_void *beta_scalar_in,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_ek_slicing,
    const int A_nthreads,
    const int A_ntasks,
    const int64_t *restrict B_ek_slicing,
    const int B_nthreads,
    const int B_ntasks
) ;

GrB_Info GB_union_jit      // C=A+B, C<#M>=A+B, eWiseUnion, via the JIT
(
    const char *kname,          // kernel base name
    // input/output:
    GrB_Matrix C,
    // input:
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GB_void *alpha_scalar_in,
    const GB_void *beta_scalar_in,
    const bool Ch_is_Mh,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const GB_task_struct *restrict TaskList,
    const int C_ntasks,
    const int C_nthreads,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_ek_slicing,
    const int A_nthreads,
    const int A_ntasks,
    const int64_t *restrict B_ek_slicing,
    const int B_nthreads,
    const int B_ntasks
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
        GB_JIT_KERNEL_UNION, false,
        false, false, C_sparsity, C->type, M, Mask_struct, Mask_comp,
        binaryop, false, A, B) ;
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
            NULL, (GB_Operator) binaryop, C->type, A->type, B->type) ;
        if (info != GrB_SUCCESS) return (info) ;

        //--------------------------------------------------------------
        // compile the jit kernel, if not found or if op/type changed
        //--------------------------------------------------------------

        if (dl_handle == NULL)
        {

            //----------------------------------------------------------
            // construct a new jit kernel for this instance
            //----------------------------------------------------------

            fprintf (fp, "\n#include \"GB_jit_kernel_ewise.h\"\n") ;
            GB_macrofy_ewise (fp, encoding.code, binaryop, C->type, A->type,
                B->type) ;
            fprintf (fp, "\n#include \"GB_jit_kernel_%s.c\"\n", kname) ;

            bool builtin = (encoding.suffix_len == 0) ;
            if (!builtin)
            {
                // create query_defn function
                GB_macrofy_query_defn (fp,
                    NULL,
                    (GB_Operator) binaryop,
                    C->type, A->type, B->type) ;
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
    info = GB_jit_kernel (C, M, A, B, alpha_scalar_in, beta_scalar_in,
        Ch_is_Mh, C_to_M, C_to_A, C_to_B, TaskList, C_ntasks, C_nthreads,
        M_ek_slicing, M_nthreads, M_ntasks, A_ek_slicing, A_nthreads, A_ntasks,
        B_ek_slicing, B_nthreads, B_ntasks) ;
    return (info) ;
#endif
}

