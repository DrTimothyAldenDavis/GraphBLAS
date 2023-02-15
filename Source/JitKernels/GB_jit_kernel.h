//------------------------------------------------------------------------------
// GB_jit_kernel.h:  JIT kernel #include for all kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd into all JIT kernels on the CPU.

#define GB_JIT_KERNEL
#include "GB.h"

#undef GB_M_TYPE
#undef GB_MCAST

// these macros are redefined by the JIT kernels to specialize them for each
// specific matrix format.

// accessing the C matrix
#undef GBP_C
#undef GBH_C
#undef GBI_C
#undef GBB_C
#undef GBX_C

// accessing the A matrix
#undef GBP_A
#undef GBH_A
#undef GBI_A
#undef GBB_A
#undef GBX_A

// accessing the B matrix
#undef GBP_B
#undef GBH_B
#undef GBI_B
#undef GBB_B
#undef GBX_B

// accessing the M matrix
#undef GBP_M
#undef GBH_M
#undef GBI_M
#undef GBB_M
#undef GBX_M

// to query the kernel for its op and type definitions:
const char *GB_jit_query_defn (int k) ;

// return true if identity and terminal match expected values
bool GB_jit_query_monoid
(
    void *id,          // input: expected identity value
    void *term,        // input: expected terminal value, if any
    size_t id_size,    // input: expected identity size
    size_t term_size   // input: expected terminal size
) ;

// return the version of SuiteSparse:GraphBLAS that created the kernel
void GB_jit_query_version
(
    int *version
) ;

