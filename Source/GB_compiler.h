//------------------------------------------------------------------------------
// GB_compiler.h: handle compiler variations
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_COMPILER_H
#define GB_COMPILER_H

//------------------------------------------------------------------------------
// determine which compiler is in use
//------------------------------------------------------------------------------

#if defined ( __INTEL_CLANG_COMPILER )

    // Intel icx compiler, 2022.0.0 based on clang/llvm 14.0.0
    #define GB_COMPILER_ICX     1
    #define GB_COMPILER_ICC     0
    #define GB_COMPILER_CLANG   0
    #define GB_COMPILER_GCC     0
    #define GB_COMPILER_MSC     0
    #define GB_COMPILER_XLC     0

    #define GB_COMPILER_MAJOR __INTEL_CLANG_COMPILER
    #define GB_COMPILER_MINOR 0
    #define GB_COMPILER_SUB   0
    #define GB_COMPILER_NAME  __VERSION__

#elif defined ( __INTEL_COMPILER )

    // Intel icc compiler: 2021.5.0 uses "gcc 7.5 mode"
    #define GB_COMPILER_ICX     0
    #define GB_COMPILER_ICC     1
    #define GB_COMPILER_CLANG   0
    #define GB_COMPILER_GCC     0
    #define GB_COMPILER_MSC     0
    #define GB_COMPILER_XLC     0

    #define GB_COMPILER_MAJOR __INTEL_COMPILER
    #define GB_COMPILER_MINOR __INTEL_COMPILER_UPDATE
    #define GB_COMPILER_SUB   0
    #define GB_COMPILER_NAME  __VERSION__

#elif defined ( __clang__ )

    // clang
    #define GB_COMPILER_ICX     0
    #define GB_COMPILER_ICC     0
    #define GB_COMPILER_CLANG   1
    #define GB_COMPILER_GCC     0
    #define GB_COMPILER_MSC     0
    #define GB_COMPILER_XLC     0

    #define GB_COMPILER_MAJOR __clang_major__
    #define GB_COMPILER_MINOR __clang_minor__
    #define GB_COMPILER_SUB   __clang_patchlevel__
    #define GB_COMPILER_NAME  "clang " __clang_version__

#elif defined ( __xlC__ )

    // xlc
    #define GB_COMPILER_ICX     0
    #define GB_COMPILER_ICC     0
    #define GB_COMPILER_CLANG   0
    #define GB_COMPILER_GCC     0
    #define GB_COMPILER_MSC     0
    #define GB_COMPILER_XLC     1

    #define GB_COMPILER_MAJOR ( __xlC__ / 256 )
    #define GB_COMPILER_MINOR ( __xlC__ - 256 * GB_COMPILER_MAJOR)
    #define GB_COMPILER_SUB   0
    #define GB_COMPILER_NAME  "IBM xlc " GB_XSTR (__xlC__)

#elif defined ( __GNUC__ )

    // gcc
    #define GB_COMPILER_ICX     0
    #define GB_COMPILER_ICC     0
    #define GB_COMPILER_CLANG   0
    #define GB_COMPILER_GCC     1
    #define GB_COMPILER_MSC     0
    #define GB_COMPILER_XLC     0

    #define GB_COMPILER_MAJOR __GNUC__
    #define GB_COMPILER_MINOR __GNUC_MINOR__
    #define GB_COMPILER_SUB   __GNUC_PATCHLEVEL__
    #define GB_COMPILER_NAME  "GNU gcc " GB_XSTR (__GNUC__) "." \
        GB_XSTR (__GNUC_MINOR__) "." GB_XSTR (__GNUC_PATCHLEVEL__)

#elif defined ( _MSC_VER )

    // Microsoft Visual Studio
    #define GB_COMPILER_ICX     0
    #define GB_COMPILER_ICC     0
    #define GB_COMPILER_CLANG   0
    #define GB_COMPILER_GCC     0
    #define GB_COMPILER_MSC     1
    #define GB_COMPILER_XLC     0

    #define GB_COMPILER_MAJOR ( _MSC_VER / 100 )
    #define GB_COMPILER_MINOR ( _MSC_VER - 100 * GB_COMPILER_MAJOR)
    #define GB_COMPILER_SUB   0
    #define GB_COMPILER_NAME  "Microsoft Visual Studio " GB_XSTR (_MSC_VER)

#else

    // other compiler
    #define GB_COMPILER_ICX     0
    #define GB_COMPILER_ICC     0
    #define GB_COMPILER_CLANG   0
    #define GB_COMPILER_GCC     0
    #define GB_COMPILER_MSC     0
    #define GB_COMPILER_XLC     0

    #define GB_COMPILER_MAJOR 0
    #define GB_COMPILER_MINOR 0
    #define GB_COMPILER_SUB   0
    #define GB_COMPILER_NAME  "other C compiler"

#endif

//------------------------------------------------------------------------------
// compiler variations
//------------------------------------------------------------------------------

// Determine the restrict keyword, and whether or not variable-length arrays
// are supported.

#if GB_COMPILER_MSC

    // Microsoft Visual Studio does not have the restrict keyword, but it does
    // support __restrict, which is equivalent.  Variable-length arrays are
    // not supported.  OpenMP tasks are not available.
    #define GB_HAS_VLA  0
    #if defined ( __cplusplus )
        // C++ does not have the restrict keyword
        #define restrict
    #else
        // C uses __restrict
        #define restrict __restrict
    #endif
    // Microsoft-specific include file
    #include <malloc.h>

#elif defined ( __cplusplus )

    #define GB_HAS_VLA  1
    // C++ does not have the restrict keyword
    #define restrict

#elif GxB_STDC_VERSION >= 199901L

    // ANSI C99 and later have the restrict keyword and variable-length arrays.
    #define GB_HAS_VLA  1

#else

    // ANSI C95 and earlier have neither
    #define GB_HAS_VLA  0
    #define restrict

#endif

//------------------------------------------------------------------------------
// PGI_COMPILER_BUG
//------------------------------------------------------------------------------

// If GraphBLAS is compiled with -DPGI_COMPILER_BUG, then a workaround is
// enabled for a bug in the PGI compiler.  The compiler does not correctly
// handle automatic arrays of variable size.

#ifdef PGI_COMPILER_BUG

    // override the ANSI C compiler to turn off variable-length arrays
    #undef  GB_HAS_VLA
    #define GB_HAS_VLA  0

#endif

//------------------------------------------------------------------------------
// OpenMP pragmas and tasks
//------------------------------------------------------------------------------

// GB_PRAGMA(x) becomes "#pragma x", but the way to do this depends on the
// compiler:
#if GB_COMPILER_MSC

    // MS Visual Studio is not ANSI C11 compliant, and uses __pragma:
    #define GB_PRAGMA(x) __pragma (x)
    // no #pragma omp simd is available in MS Visual Studio
    #define GB_PRAGMA_SIMD
    #define GB_PRAGMA_SIMD_REDUCTION(op,s)

#else

    // ANSI C11 compilers use _Pragma:
    #define GB_PRAGMA(x) _Pragma (#x)
    // create two kinds of SIMD pragmas:
    // GB_PRAGMA_SIMD becomes "#pragma omp simd"
    // GB_PRAGMA_SIMD_REDUCTION (+,cij) becomes
    // "#pragma omp simd reduction(+:cij)"
    #define GB_PRAGMA_SIMD GB_PRAGMA (omp simd)
    #define GB_PRAGMA_SIMD_REDUCTION(op,s) GB_PRAGMA (omp simd reduction(op:s))

#endif

#define GB_PRAGMA_IVDEP GB_PRAGMA(ivdep)

//------------------------------------------------------------------------------
// variable-length arrays
//------------------------------------------------------------------------------

// If variable-length arrays are not supported, user-defined types are limited
// in size to 128 bytes or less.  Many of the type-generic routines allocate
// workspace for a single scalar of variable size, using a statement:
//
//      GB_void aij [xsize] ;
//
// To support non-variable-length arrays in ANSI C95 or earlier, this is used:
//
//      GB_void aij [GB_VLA(xsize)] ;
//
// GB_VLA(xsize) is either defined as xsize (for ANSI C99 or later), or a fixed
// size of 128, in which case user-defined types
// are limited to a max of 128 bytes.

#if ( GB_HAS_VLA )

    // variable-length arrays are allowed
    #define GB_VLA(s) s

#else

    // variable-length arrays are not allowed
    #define GB_VLA_MAXSIZE 128
    #define GB_VLA(s) GB_VLA_MAXSIZE

#endif

//------------------------------------------------------------------------------
// AVX2 and AVX512F support for the x86_64 architecture
//------------------------------------------------------------------------------

// gcc 7.5.0 cannot compile code with __attribute__ ((target ("avx512f"))), or
// avx2 (it triggers a bug in the compiler), but those targets are fine with
// gcc 9.3.0 or later.  It might be OK on gcc 8.x but I haven't tested this.

#if GBX86

    #if GB_COMPILER_GCC
        #if __GNUC__ >= 9
            // enable avx512f on gcc 9.x and later
            #define GB_COMPILER_SUPPORTS_AVX512F 1
            #define GB_COMPILER_SUPPORTS_AVX2 1
        #else
            // disable avx2 and avx512f on gcc 8.x and earlier
            #define GB_COMPILER_SUPPORTS_AVX512F 0
            #define GB_COMPILER_SUPPORTS_AVX2 0
        #endif
    #elif GB_COMPILER_ICX || GB_COMPILER_ICC || GB_COMPILER_CLANG || \
          GB_COMPILER_GCC || GB_COMPILER_MSC
        // all these compilers can handle AVX512F and AVX2 on x86
        #define GB_COMPILER_SUPPORTS_AVX512F 1
        #define GB_COMPILER_SUPPORTS_AVX2 1
    #else
        // unsure if xlc can handle AVX, but it is not likely to be used on
        // the x86 anyay
        #define GB_COMPILER_SUPPORTS_AVX512F 0
        #define GB_COMPILER_SUPPORTS_AVX2 0
    #endif

#else

    // non-X86_64 architecture
    #define GB_COMPILER_SUPPORTS_AVX512F 0
    #define GB_COMPILER_SUPPORTS_AVX2 0

#endif

// prefix for function with target avx512f
#if GB_COMPILER_SUPPORTS_AVX512F
    #if GB_COMPILER_MSC
        #define GB_TARGET_AVX512F __declspec (target ("avx512f"))
    #else
        #define GB_TARGET_AVX512F __attribute__ ((target ("avx512f")))
    #endif
#else
#define GB_TARGET_AVX512F
#endif

// prefix for function with target avx2
#if GB_COMPILER_SUPPORTS_AVX2
    #if GB_COMPILER_MSC
        #define GB_TARGET_AVX2 __declspec (target ("avx2"))
    #else
        #define GB_TARGET_AVX2 __attribute__ ((target ("avx2")))
    #endif
#else
#define GB_TARGET_AVX2
#endif

#endif

