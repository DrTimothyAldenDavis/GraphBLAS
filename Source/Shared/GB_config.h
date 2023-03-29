//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_config.h: JIT configuration for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is configured by cmake.

#ifndef GB_CONFIG_H
#define GB_CONFIG_H

// GB_C_COMPILER: the C compiler used to compile GraphBLAS:
#define GB_C_COMPILER   "/usr/bin/clang"

// GB_C_FLAGS: the C compiler flags used to compile GraphBLAS.  Used
// for compiling and linking:
#define GB_C_FLAGS      " -Wno-pointer-sign  -O3 -DNDEBUG -fopenmp=libomp  -fPIC "

// GB_C_LINK_FLAGS: the flags passed to the C compiler for the link phase:
#define GB_C_LINK_FLAGS " -shared "

// GB_LIB_SUFFIX: library suffix (.so for Linux/Unix, .dylib for Mac, etc):
#define GB_LIB_SUFFIX   ".so"

// GB_OBJ_SUFFIX: object suffix (.o for Linux/Unix/Mac, .obj for Windows):
#define GB_OBJ_SUFFIX   ".o"

// GB_SOURCE_PATH: the source code for GraphBLAS, which is the path of the
// top-level GraphBLAS folder:
#define GB_SOURCE_PATH  "/home/faculty/d/davis/master/GraphBLAS"

// GB_BUILD_PATH: the location where GraphBLAS was built.  This is only used
// if the GraphBLAS cache path cannot be determined by GrB_init.
#define GB_BUILD_PATH   "/home/faculty/d/davis/master/GraphBLAS/build"

// GB_OMP_INC: include directories for OpenMP, if in use by GraphBLAS:
#define GB_OMP_INC      ""

// GB_LIBRARIES: libraries to link with
#define GB_LIBRARIES    " -lm -ldl /usr/lib/llvm-10/lib/libomp.so /usr/lib/x86_64-linux-gnu/libpthread.so"
#endif

