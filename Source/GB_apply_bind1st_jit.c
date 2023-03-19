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
    const char *kname,          // kernel base name
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

    //--------------------------------------------------------------------------
    // enumify the problem and look it up in the jit hash
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GBURBLE ("(jit) ") ;
    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_KERNEL_APPLYBIND1, false,
        false, false, GxB_FULL, ctype, NULL, false, false,
        binaryop, false, NULL, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    info = GB_jitifyer_load (&dl_function, GB_jit_ewise_family, kname,
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, ctype, NULL, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cx, xscalar, B->x, B->b, GB_nnz_held (B),
        nthreads)) ;
#endif
}

