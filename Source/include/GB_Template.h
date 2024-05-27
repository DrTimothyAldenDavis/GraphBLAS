//------------------------------------------------------------------------------
// GB_Template.h: internal definitions for GraphBLAS, including JIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_TEMPLATE_H
#define GB_TEMPLATE_H

//------------------------------------------------------------------------------
// definitions that modify GraphBLAS.h
//------------------------------------------------------------------------------

#include "include/GB_dev.h"
#include "include/GB_compiler.h"
#include "include/GB_warnings.h"
#include "include/GB_coverage.h"

//------------------------------------------------------------------------------
// user-visible GraphBLAS.h
//------------------------------------------------------------------------------

#include "GraphBLAS.h"
#undef I

#ifdef GBMATLAB
#undef GRAPHBLAS_HAS_CUDA
#endif

//------------------------------------------------------------------------------
// handle the restrict and 'static inline' keywords
//------------------------------------------------------------------------------

// Intentionally shadow the built-in "restrict" keyword.  See GraphBLAS.h for
// the definition of GB_restrict.  It becomes empty for C++, and "__restrict"
// for MS Visual Studio.  Otherwise, GB_restrict is just "restrict" on C11
// compliant compilers.  I prefer to use the "restrict" keyword to make the
// code readable.  This #define is a patch for compilers that don't support it:

#define restrict GB_restrict

// for internal static inline functions (will be redefined for CUDA)
#undef  GB_STATIC_INLINE
#define GB_STATIC_INLINE static inline

//------------------------------------------------------------------------------
// internal #include files
//------------------------------------------------------------------------------

#include "shared/GB_index.h"
#include "shared/GB_complex.h"
#include "shared/GB_pun.h"
#include "shared/GB_opaque.h"
#include "shared/GB_partition.h"
#include "shared/GB_hash.h"
#include "shared/GB_int64_mult.h"

#include "include/GB_prefix.h"
#include "include/GB_defaults.h"
#include "include/GB_binary_search.h"
#include "include/GB_zombie.h"

#ifdef GB_JIT_KERNEL

    #include "include/GB_bytes.h"
    #include "include/GB_atomics.h"
    #include "include/GB_printf_kernels.h"
    #include "include/GB_assert_kernels.h"
    #include "include/GB_math_macros.h"
    #include "include/GB_iceil.h"
    #include "include/GB_memory_macros.h"
    #include "include/GB_werk.h"
    #include "include/GB_nthreads.h"
    #include "include/GB_log2.h"
    #include "include/GB_task_struct.h"
    #include "include/GB_wait_macros.h"
    #include "include/GB_AxB_macros.h"
    #include "include/GB_ek_slice_kernels.h"
    #include "include/GB_bitmap_scatter.h"
    #include "include/GB_omp_kernels.h"
    #include "include/GB_callback_proto.h"
    #include "include/GB_saxpy3task_struct.h"
    #include "include/GB_callback.h"

#else

    #include "type/include/GB_bytes.h"
    #include "omp/include/GB_atomics.h"
    #include "print/include/GB_printf_kernels.h"
    #include "ok/include/GB_assert_kernels.h"
    #include "math/include/GB_math_macros.h"
    #include "math/include/GB_iceil.h"
    #include "memory/include/GB_memory_macros.h"
    #include "werk/include/GB_werk.h"
    #include "omp/include/GB_nthreads.h"
    #include "math/include/GB_log2.h"
    #include "slice/include/GB_task_struct.h"
    #include "wait/include/GB_wait_macros.h"
    #include "mxm/include/GB_AxB_macros.h"
    #include "slice/include/GB_ek_slice_kernels.h"
    #include "assign/include/GB_bitmap_scatter.h"
    #include "omp/include/GB_omp_kernels.h"
    #include "callback/include/GB_callback_proto.h"
    #include "mxm/include/GB_saxpy3task_struct.h"
    #include "callback/include/GB_callback.h"

#endif

#include "shared/GB_hyper_hash_lookup.h"

#endif

