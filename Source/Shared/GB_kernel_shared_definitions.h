//------------------------------------------------------------------------------
// GB_kernel_shared_definitions.h: definitions for all methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This header is #include'd just before using any templates for any method:
// pre-generated kernel, CPU or GPU JIT, or generic.

#ifndef GB_KERNEL_SHARED_DEFINITIONS_H
#define GB_KERNEL_SHARED_DEFINITIONS_H

//------------------------------------------------------------------------------
// atomic compare/exchange for the GB_Z_TYPE data type
//------------------------------------------------------------------------------

#if defined ( GB_GENERIC ) || !defined ( GB_Z_ATOMIC_BITS ) || defined ( GB_CUDA_KERNEL )

    //--------------------------------------------------------------------------
    // no atomic compare/exchange
    //--------------------------------------------------------------------------

    // Attempting to use the atomic compare/exchange will generate an
    // intentional compile-time error.

    // Generic kernels cannot use a single atomic compare/exchange method
    // determined at compile time.  They would need to use a run-time
    // selection, based on zsize (not currently used).

    // If this file is #include'd in a CUDA kernel, the atomics must use the
    // atomicCAS and other methods, #define'd in GB_cuda_atomics.cuh.  This
    // method is not used.

    // If GB_Z_ATOMIC_BITS is not #define'd, then the kernel does not have a
    // GB_Z_TYPE, or it's not the correct size to use an atomic
    // compare/exchange.

    #define GB_HAS_ATOMIC_COMPARE_EXCHANGE 0
    #define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) none

#else

    //--------------------------------------------------------------------------
    // atomic compare/exchange for 0, 8, 16, 32, and 64-bit data types
    //--------------------------------------------------------------------------

    // The CPU JIT kernels can use these kernels for user-defined types of
    // the right size.

    #if ( GB_Z_ATOMIC_BITS == 0 )

        // No atomic compare/exchange needed (the ANY monoid). This is a no-op.
        #define GB_HAS_ATOMIC_COMPARE_EXCHANGE 1
        #define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired)

    #elif ( GB_Z_ATOMIC_BITS == 8 )

        // atomic compare/exchange for int8_t, uint8_t
        #define GB_HAS_ATOMIC_COMPARE_EXCHANGE 1
        #define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
              GB_ATOMIC_COMPARE_EXCHANGE_8(target, expected, desired)

    #elif ( GB_Z_ATOMIC_BITS == 16 )

        // atomic compare/exchange for int16_t, uint16_t
        #define GB_HAS_ATOMIC_COMPARE_EXCHANGE 1
        #define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
             GB_ATOMIC_COMPARE_EXCHANGE_16(target, expected, desired)

    #elif ( GB_Z_ATOMIC_BITS == 32 )

        // atomic compare/exchange for int32_t, uint32_t, and float
        #define GB_HAS_ATOMIC_COMPARE_EXCHANGE 1
        #define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
             GB_ATOMIC_COMPARE_EXCHANGE_32(target, expected, desired)

    #elif ( GB_Z_ATOMIC_BITS == 64 )

        // atomic compare/exchange for int64_t, uint64_t, double,
        // and float complex
        #define GB_HAS_ATOMIC_COMPARE_EXCHANGE 1
        #define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) \
             GB_ATOMIC_COMPARE_EXCHANGE_64(target, expected, desired)

    #else

        // no atomic compare/exchange available (compile-time error)
        #define GB_HAS_ATOMIC_COMPARE_EXCHANGE 0
        #define GB_ATOMIC_COMPARE_EXCHANGE(target, expected, desired) none

    #endif

#endif
#endif

