#-------------------------------------------------------------------------------
# GraphBLAS/JITpackage/CMakeLists.txt:  package source for the GraphBLAS JIT
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required ( VERSION 3.19 )

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../cmake_modules )

include ( GraphBLAS_version )

project ( grb_jitpackage
    VERSION "${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}"
    LANGUAGES C )

include_directories ( ../zstd ../zstd/zstd_subset )

file ( GLOB GRB_PACKAGE_SOURCES "grb_jitpackage.c" )

add_executable ( grb_jitpackage  "grb_jitpackage.c" )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( grb_jitpackage PUBLIC m )
endif ( )

