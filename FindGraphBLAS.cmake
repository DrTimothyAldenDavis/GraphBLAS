#-------------------------------------------------------------------------------
# GraphBLAS/cmake/FindGraphBLAS.cmake
#-------------------------------------------------------------------------------

# This FindGraphBLAS.cmake file is copied from LAGraph,
# (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Find the native GRAPHBLAS includes and library.

# IMPORTED Targets:
# This module defines :prop_tgt:`IMPORTED` target ``GRAPHBLAS::GRAPHBLAS``, if
# GRAPHBLAS has been found.

# Result Variables: This module defines the following variables:

#   GRAPHBLAS_INCLUDE_DIR    - where to find GraphBLAS.h, etc.
#   GRAPHBLAS_LIBRARY        - GraphBLAS library
#   GRAPHBLAS_LIBRARIES      - List of libraries when using GraphBLAS.
#   GRAPHBLAS_FOUND          - True if GraphBLAS found.

# Hints:

# A user may set ``GRAPHBLAS_ROOT`` to a GraphBLAS installation root to tell
# this module where to look.

# Otherwise, the first place searched is in ../GraphBLAS, relative to the
# current source directory.  That is, if GraphBLAS and the user application
# (such as LAGraph) reside in the same parent folder, side-by-side, and if it
# contains GraphBLAS/Include/GraphBLAS.h file and
# GraphBLAS/build/libgraphblas.so (or dylib, etc), then that version is used.
# This takes precedence over the system-wide installation of GraphBLAS, which
# might be an older version.  This method gives the user the ability to compile
# their application with their own copy of GraphBLAS, ignoring the system-wide
# version.

# To use this file in your application, place in MyApp/cmake, where MyApp
# is your application.

#-------------------------------------------------------------------------------

# "Include" for SuiteSparse:GraphBLAS
find_path(
  GRAPHBLAS_INCLUDE_DIR
  NAMES GraphBLAS.h
  HINTS ${CMAKE_SOURCE_DIR}/../GraphBLAS
  PATHS GRAPHBLAS_ROOT ENV GRAPHBLAS_ROOT
  PATH_SUFFIXES include Include
  )

# "build" for SuiteSparse:GraphBLAS
find_library(
  GRAPHBLAS_LIBRARY
  NAMES graphblas
  HINTS ${CMAKE_SOURCE_DIR}/../GraphBLAS
  PATHS GRAPHBLAS_ROOT ENV GRAPHBLAS_ROOT
  PATH_SUFFIXES lib build alternative
  )

# get version of .so using REALPATH
get_filename_component(GRAPHBLAS_LIBRARY ${GRAPHBLAS_LIBRARY} REALPATH)
string(
  REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
  GRAPHBLAS_VERSION
  ${GRAPHBLAS_LIBRARY}
  )
set(GRAPHBLAS_LIBRARIES ${GRAPHBLAS_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  GraphBLAS
  REQUIRED_VARS GRAPHBLAS_LIBRARIES GRAPHBLAS_INCLUDE_DIR
  VERSION_VAR GRAPHBLAS_VERSION
  )

mark_as_advanced(
  GRAPHBLAS_INCLUDE_DIR
  GRAPHBLAS_LIBRARY
  GRAPHBLAS_LIBRARIES
  )

if ( GRAPHBLAS_FOUND )
    message ( STATUS "GraphBLAS include dir: " ${GRAPHBLAS_INCLUDE_DIR} )
    message ( STATUS "GraphBLAS library:     " ${GRAPHBLAS_LIBRARY} )
    message ( STATUS "GraphBLAS version:     " ${GRAPHBLAS_VERSION} )
else ( )
    message ( STATUS "GraphBLAS not found" )
endif ( )

