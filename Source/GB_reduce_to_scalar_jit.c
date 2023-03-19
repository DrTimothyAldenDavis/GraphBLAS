//------------------------------------------------------------------------------
// GB_reduce_to_scalar_jit: reduce a matrix to a scalar, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_reduce.h"
#include "GB_binop.h"
#include "GB_stringify.h"
#include "GB_jitifyer.h"

typedef GrB_Info (*GB_jit_dl_function)
(
    void *,
    const GrB_Matrix,
    void *,
    bool *,
    int,
    int
) ;

GrB_Info GB_reduce_to_scalar_jit    // z = reduce_to_scalar (A) via the JIT
(
    const char *kname,          // kernel base name
    // output:
    void *z,                    // result
    // input:
    const GrB_Monoid monoid,    // monoid to do the reduction
    const GrB_Matrix A,         // matrix to reduce
    GB_void *restrict W,        // workspace
    bool *restrict F,           // workspace
    int ntasks,                 // # of tasks to use
    int nthreads                // # of threads to use
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
    uint64_t hash = GB_encodify_reduce (&encoding, &suffix, monoid, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function, GB_jit_reduce_family, kname,
        hash, &encoding, suffix, NULL, monoid,
        NULL, A->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (z, A, W, F, ntasks, nthreads)) ;
#endif
}

