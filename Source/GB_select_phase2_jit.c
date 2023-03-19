//------------------------------------------------------------------------------
// GB_select_phase2_jit: select phase 2 for the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_select.h"
#include "GB_stringify.h"
#include "GB_jitifyer.h"

typedef GrB_Info (*GB_jit_dl_function)
(
    int64_t *restrict Ci,
    GB_void *restrict Cx,                   // NULL if C is iso-valued
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_select_phase2_jit      // select phase2
(
    const char *kname,          // kernel base name
    // output:
    int64_t *restrict Ci,
    GB_void *restrict Cx,                   // NULL if C is iso-valued
    // input:
    const int64_t *restrict Cp,
    const bool C_iso,
    const bool in_place_A,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
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
    uint64_t hash = GB_encodify_select (&encoding, &suffix,
        GB_JIT_KERNEL_SELECT2, C_iso, in_place_A, op, flipij, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function, GB_jit_select_family, kname,
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) op, A->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Ci, Cx, Cp, Cp_kfirst, A, ythunk, A_ek_slicing,
        A_ntasks, A_nthreads)) ;
#endif
}

