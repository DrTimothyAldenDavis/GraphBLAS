//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_config.h: JIT configuration for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_BUILD_PATH: the location where GraphBLAS was built.  This is only used
// if the GraphBLAS cache path cannot be determined by GrB_init.

// GB_SOURCE_PATH: the source code for GraphBLAS, which is the path of the
// top-level GraphBLAS folder.

// GB_C_COMPILER: the C compiler used to compile GraphBLAS

// GB_C_FLAGS: the C compiler flags used to compile GraphBLAS

// This file is configured by cmake.

#ifndef GB_CONFIG_H
#define GB_CONFIG_H

#define GB_BUILD_PATH  "/home/davis/master/GraphBLAS/build"
#define GB_SOURCE_PATH "/opt/SuiteSparse/GraphBLAS"
#define GB_C_COMPILER  "/usr/bin/gcc"
#define GB_C_FLAGS     " -std=c11 -lm -Wno-pragmas  -fexcess-precision=fast  -fcx-limited-range  -fno-math-errno  -fwrapv  -g -fopenmp "

#endif

