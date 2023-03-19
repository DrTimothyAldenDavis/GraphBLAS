//------------------------------------------------------------------------------
// GB_AxB_saxpy3_jit: C<M>=A*B saxpy3 method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_stringify.h"
#include "GB_jitifyer.h"

typedef GrB_Info (*GB_jit_dl_function)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int ntasks,
    const int nfine,
    const int nthreads,
    const int do_sort,
    GB_Werk
) ;

GrB_Info GB_AxB_saxpy3_jit      // C<M>=A*B, saxpy3, via the JIT
(
    const char *kname,          // kernel base name
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    void *SaxpyTasks,
    const int ntasks,
    const int nfine,
    const int nthreads,
    const int do_sort,          // if nonzero, try to sort in saxpy3
    GB_Werk Werk
)
{

#ifdef GBRENAME
    return (GrB_NO_VALUE) ;
#else

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_mxm (&encoding, &suffix,
        GB_JIT_KERNEL_AXB_SAXPY3,
        C->iso, false, GB_sparsity (C), C->type,
        M, Mask_struct, Mask_comp, semiring, flipxy, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function, GB_jit_mxm_family, kname,
        hash, &encoding, suffix, semiring, NULL,
        NULL, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, M, M_in_place, A, B,
        (GB_saxpy3task_struct *) SaxpyTasks, ntasks, nfine, nthreads, do_sort,
        Werk)) ;
#endif
}

