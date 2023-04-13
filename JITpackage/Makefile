#-------------------------------------------------------------------------------
# GraphBLAS/Package/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

default: library

library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . -j${JOBS} )

clean: distclean

purge: distclean

# remove all files not in the distribution
distclean:
	- rm -rf build/* GB_JITpackage*

