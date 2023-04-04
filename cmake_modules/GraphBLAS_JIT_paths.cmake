#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS_JIT_paths.cmake:  configure the JIT paths
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

#-------------------------------------------------------------------------------
# define the source and cache paths
#-------------------------------------------------------------------------------

# The GraphBLAS CPU and CUDA JITs need to know where the GraphBLAS source is
# located, and where to put the compiled libraries.

include ( SuiteSparse_getenv )

# set the GRAPHBLAS_SOURCE_PATH
if ( DEFINED GRAPHBLAS_SOURCE_PATH )
    # if the -DGRAPHBLAS_SOURCE_PATH variable is set use that setting
elseif ( DEFINED ENV{GRAPHBLAS_SOURCE_PATH} )
    # otherwise, use the GRAPHBLAS_SOURCE_PATH environment variable
    set ( GRAPHBLAS_SOURCE_PATH "$ENV{GRAPHBLAS_SOURCE_PATH}" )
else ( )
    # if no other option, use the CMAKE_SOURCE_DIR.  This is the typical
    # case if the one compiling GraphBLAS does not set any variables first.
    set ( GRAPHBLAS_SOURCE_PATH ${CMAKE_SOURCE_DIR} )
endif ( )

# set the GRAPHBLAS_BUILD_PATH
if ( DEFINED GRAPHBLAS_BUILD_PATH )
    # if the -DGRAPHBLAS_BUILD_PATH variable is set use that setting
elseif ( DEFINED ENV{GRAPHBLAS_BUILD_PATH} )
    # otherwise, use the GRAPHBLAS_BUILD_PATH environment variable
    set ( GRAPHBLAS_BUILD_PATH "$ENV{GRAPHBLAS_BUILD_PATH}" )
else ( )
    # if no other option, use the CMAKE_BINARY_DIR.  This is the typical
    # case if the one compiling GraphBLAS does not set any variables first.
    set ( GRAPHBLAS_BUILD_PATH ${CMAKE_BINARY_DIR} )
endif ( )

# set the GRAPHBLAS_CACHE_PATH for compiled JIT kernels
if ( DEFINED ENV{GRAPHBLAS_CACHE_PATH} )
    # use the GRAPHBLAS_CACHE_PATH environment variable
    set ( GRAPHBLAS_CACHE_PATH "$ENV{GRAPHBLAS_CACHE_PATH}" )
elseif ( DEFINED ENV{HOME} )
    # use the current HOME environment variable from cmake (for Linux, Unix, Mac)
    set ( GRAPHBLAS_CACHE_PATH "$ENV{HOME}/.SuiteSparse/GraphBLAS/${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}" )
elseif ( WIN32 )
    # use LOCALAPPDATA for Windows
    set ( GRAPHBLAS_CACHE_PATH "$ENV{LOCALAPPDATA}/SuiteSparse/GraphBLAS/${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}" )
else ( )
    # if no other option, the cache path must be set at run time via GxB_set
    set ( GRAPHBLAS_CACHE_PATH "cache path not found" )
endif ( )

# set the GRAPHBLAS_MATLAB_PATH for compiled JIT kernels for MATLAB
if ( DEFINED ENV{GRAPHBLAS_MATLAB_PATH} )
    # use the GRAPHBLAS_MATLAB_PATH environment variable
    set ( GRAPHBLAS_MATLAB_PATH "$ENV{GRAPHBLAS_MATLAB_PATH}" )
elseif ( DEFINED ENV{HOME} )
    # use the current HOME environment variable from cmake (for Linux, Unix, Mac)
    set ( GRAPHBLAS_MATLAB_PATH "$ENV{HOME}/.SuiteSparse/GraphBLAS/${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}_matlab" )
elseif ( WIN32 )
    # use LOCALAPPDATA for Windows
    set ( GRAPHBLAS_MATLAB_PATH "$ENV{LOCALAPPDATA}/SuiteSparse/GraphBLAS/${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}_matlab" )
else ( )
    # if no other option, the cache path must be set at run time via GxB_set
    set ( GRAPHBLAS_MATLAB_PATH "cache path not found" )
endif ( )

#-------------------------------------------------------------------------------
# NJIT and COMPACT options
#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    # FOR NOW: do not compile FactoryKernels when developing the CUDA kernels
    set ( COMPACT on )
endif ( )

option ( COMPACT "ON: do not compile FactoryKernels.  OFF (default): compile FactoryKernels" off )
option ( NJIT "ON: do not use the CPU JIT.  OFF (default): enable the CPU JIT" off )

if ( NJIT )
    # disable the CPU JIT (but keep any PreJIT kernels enabled)
    add_compile_definitions ( NJIT )
    message ( STATUS "GraphBLAS CPU JIT: disabled (any PreJIT kernels will still be enabled)")
else ( )
    message ( STATUS "GraphBLAS CPU JIT: enabled")
endif ( )

if ( COMPACT )
    add_compile_definitions ( GBCOMPACT )
    message ( STATUS "GBCOMPACT: enabled; FactoryKernels will not be built" )
endif ( )

