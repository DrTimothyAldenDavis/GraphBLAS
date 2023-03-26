//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_config.h: JIT configuration for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_C_COMPILER: the C compiler used to compile GraphBLAS

// GB_C_FLAGS: the C compiler flags used to compile GraphBLAS.  Used
// for compiling and linking.

// GB_C_LINK_FLAGS: the flags passed to the C compiler for the link phase.

// GB_SOURCE_PATH: the source code for GraphBLAS, which is the path of the
// top-level GraphBLAS folder.

// GB_BUILD_PATH: the location where GraphBLAS was built.  This is only used
// if the GraphBLAS cache path cannot be determined by GrB_init.

// This file is configured by cmake.

#ifndef GB_CONFIG_H
#define GB_CONFIG_H

#define GB_C_COMPILER   "/opt/homebrew/bin/gcc-12"
#define GB_C_FLAGS      " -std=c11 -lm -Wno-pragmas  -fexcess-precision=fast  -fcx-limited-range  -fno-math-errno  -fwrapv  -O3 -DNDEBUG -fopenmp  -fPIC  -arch arm64  -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX13.1.sdk "
#define GB_C_LINK_FLAGS " -dynamiclib "
#define GB_LIB_SUFFIX   ".dylib"
#define GB_OBJ_SUFFIX   ".o"
#define GB_SOURCE_PATH  "/Users/davis/master/GraphBLAS"
#define GB_BUILD_PATH   "/Users/davis/master/GraphBLAS/build"

#endif

