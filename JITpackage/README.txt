GraphBLAS/JITPackage:  package GraphBLAS source for the JIT 

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

The use of this package is not required by the end user.  If you edit the
GraphBLAS source code itself, however, you need to run "make" in this
directory to update the GB_JITpackage.c file before compiling GraphBLAS.

This small stand-alone package compresses all the source files (*.c and *.h)
required by the JIT kernels into a single file: GB_JITpackage.c.  When
GraphBLAS is prepared for distribution, a "make" in this directory updates the
GB_JITpackage.c file.  When GraphBLAS is compiled via cmake, it compiles
GB_JITpackage.c into the libgraphblas.so (or dylib, dll, whatever).

When GraphBLAS starts, GrB_init checks the user source folder to ensure
~/.SuiteSparse/GrBx.y.z/src exists (where x.y.z is the current GraphBLAS
version number), and that it contains all the GraphBLAS source code.  If not,
it uncompresses each file from its compressed form in GB_JITpackage.c, and
writes it to the user source folder.

To create the GB_JITpackage.c file without "make", do the following:

    cd build
    cmake ..
    cmake --build .

