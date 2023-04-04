GraphBLAS/GraphBLAS/PreJIT:  This folder is empty in the GraphBLAS distribution.

JIT kernel source files created by GraphBLAS in MATLAB may be placed in this
folder by the end user, by copying them from your
~/.SuiteSparse/GraphBLAS/*.*.*/MATLAB folder.

If GraphBLAS is then recompiled via cmake, the build system will compile all of
these kernels and make them available as 'pre-compiled JIT kernels'.  The
kernels are no longer JIT kernels since they are not compiled at run-time, but
they are still refered to as "JIT" kernels since they were at one time created
at run time by the GraphBLAS JIT.  Thus the name of this folder:
GraphBLAS/GraphBLAS/PreJIT.

If the GraphBLAS version is changed at all (even in the last digit), all *.c
files in this folder must be deleted.

If a user-defined type or operator is changed, the relevant kernels should be
be deleted.

