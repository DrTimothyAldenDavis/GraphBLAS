#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS_JIT_paths.cmake:  configure the JIT paths
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

#-------------------------------------------------------------------------------

# construct the JIT compiler/link strings
set ( GB_SOURCE_PATH "${GRAPHBLAS_SOURCE_PATH}" )
set ( GB_BUILD_PATH  "${GRAPHBLAS_BUILD_PATH}" )
set ( GB_C_COMPILER  "${CMAKE_C_COMPILER}" )
set ( GB_C_FLAGS "${CMAKE_C_FLAGS}" )
set ( GB_C_LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
set ( GB_LIB_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}" )

# construct the C flags and link flags
if ( APPLE )
    # MacOS
    set ( GB_C_FLAGS "${GB_C_FLAGS} -fPIC " )
    set ( GB_C_FLAGS "${GB_C_FLAGS} -arch ${CMAKE_HOST_SYSTEM_PROCESSOR} " )
    set ( GB_C_FLAGS "${GB_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT} " )
    set ( GB_C_LINK_FLAGS "${GB_C_LINK_FLAGS} -dynamiclib " )
    set ( GB_OBJ_SUFFIX ".o" )
elseif ( WIN32 )
    # Windows
    # FIXME: need more here for Windows
    set ( GB_OBJ_SUFFIX ".dll" )
else ( )
    # Linux / Unix
    set ( GB_C_FLAGS "${GB_C_FLAGS} -fPIC " )
    set ( GB_C_LINK_FLAGS "${GB_C_LINK_FLAGS} -shared " )
    set ( GB_OBJ_SUFFIX ".o" )
endif ( )

# construct the -I list for OpenMP
if ( OPENMP_FOUND )
    set ( GB_OMP_INC ${OpenMP_C_INCLUDE_DIRS} )
    list ( TRANSFORM GB_OMP_INC PREPEND " -I" )
else ( )
    set ( GB_OMP_INC "" )
endif ( )

# construct the library list
string ( REPLACE "." "\\." LIBSUFFIX ${GB_LIB_SUFFIX} )
set ( GB_LIBRARIES "" )
#   message ( STATUS "lib suffix: ${LIBSUFFIX}" )
foreach ( LIB_NAME ${GB_LIST_LIB} )
#       message ( STATUS "Lib: ${LIB_NAME}" )
    if ( LIB_NAME MATCHES ${LIBSUFFIX} )
#           message ( STATUS "has suffix" )
        string ( APPEND GB_LIBRARIES " " ${LIB_NAME} )
    else ( )
#           message ( STATUS "no suffix" )
        string ( APPEND GB_LIBRARIES " -l" ${LIB_NAME} )
    endif ( )
endforeach ( )

if ( NOT NJIT OR ENABLE_CUDA )
    message ( STATUS "------------------------------------------------------------------------" )
    message ( STATUS "JIT configuration:" )
    message ( STATUS "------------------------------------------------------------------------" )
    # one or both JITs are enabled; make sure the cache path exists
    message ( STATUS "JIT C compiler: ${GB_C_COMPILER}" )
    message ( STATUS "JIT C flags:    ${GB_C_FLAGS}" )
    message ( STATUS "JIT link flags: ${GB_C_LINK_FLAGS}" )
    message ( STATUS "JIT lib suffix: ${GB_LIB_SUFFIX}" )
    message ( STATUS "JIT obj suffix: ${GB_OBJ_SUFFIX}" )
    message ( STATUS "JIT source:     ${GB_SOURCE_PATH}" )
    message ( STATUS "JIT cache:      ${GRAPHBLAS_CACHE_PATH}" )
    message ( STATUS "JIT fall-back:  ${GB_BUILD_PATH}" )
    message ( STATUS "JIT openmp inc: ${GB_OMP_INC}" )
    message ( STATUS "JIT libraries:  ${GB_LIBRARIES}" )
endif ( )

# create a list of files
file ( GLOB PRE1 "PreJIT/GB_jit_*.c" )
set ( PREJIT "" )
set ( PREPRO "" )
set ( PREQUERY "" )
set ( PREQ "" )
foreach ( PSRC ${PRE1} )
    get_filename_component ( F ${PSRC} NAME_WE )
    list ( APPEND PREJIT ${F} )
    list ( APPEND PREQUERY "JIT_Q (" ${F} "_query)\n" )
    list ( APPEND PREQ "${F}_query" )
    if ( ${F} MATCHES "^GB_jit__add_" )
        list ( APPEND PREPRO "JIT_ADD  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__apply_bind1" )
        list ( APPEND PREPRO "JIT_AP1  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__apply_bind2" )
        list ( APPEND PREPRO "JIT_AP2  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__apply_unop" )
        list ( APPEND PREPRO "JIT_AP0  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_dot2_" )
        list ( APPEND PREPRO "JIT_DOT2 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_dot2n_" )
        list ( APPEND PREPRO "JIT_DO2N (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_dot3_" )
        list ( APPEND PREPRO "JIT_DOT3 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_dot4_" )
        list ( APPEND PREPRO "JIT_DOT4 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_saxbit" )
        list ( APPEND PREPRO "JIT_SAXB (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_saxpy3" )
        list ( APPEND PREPRO "JIT_SAX3 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_saxpy4" )
        list ( APPEND PREPRO "JIT_SAX4 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_saxpy5" )
        list ( APPEND PREPRO "JIT_SAX5 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__build" )
        list ( APPEND PREPRO "JIT_BLD  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__colscale" )
        list ( APPEND PREPRO "JIT_COLS (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__concat_bitmap" )
        list ( APPEND PREPRO "JIT_CONB (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__concat_full" )
        list ( APPEND PREPRO "JIT_CONF (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__concat_sparse" )
        list ( APPEND PREPRO "JIT_CONS (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__convert_s2b" )
        list ( APPEND PREPRO "JIT_CS2B (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_02" )
        list ( APPEND PREPRO "JIT_EM2  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_03" )
        list ( APPEND PREPRO "JIT_EM3  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_04" )
        list ( APPEND PREPRO "JIT_EM4  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_08" )
        list ( APPEND PREPRO "JIT_EM8  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_bitmap" )
        list ( APPEND PREPRO "JIT_EMB  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__ewise_full_accum" )
        list ( APPEND PREPRO "JIT_EWFA (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__ewise_full_noaccum" )
        list ( APPEND PREPRO "JIT_EWFN (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__reduce" )
        list ( APPEND PREPRO "JIT_RED  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__rowscale" )
        list ( APPEND PREPRO "JIT_ROWS (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__select_bitmap" )
        list ( APPEND PREPRO "JIT_SELB (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__select_phase1" )
        list ( APPEND PREPRO "JIT_SEL1 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__select_phase2" )
        list ( APPEND PREPRO "JIT_SEL2 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__split_bitmap" )
        list ( APPEND PREPRO "JIT_SPB  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__split_full" )
        list ( APPEND PREPRO "JIT_SPF  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__split_sparse" )
        list ( APPEND PREPRO "JIT_SPS  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__subassign" )
        list ( APPEND PREPRO "JIT_SUB  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__trans_bind1" )
        list ( APPEND PREPRO "JIT_TR1  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__trans_bind2" )
        list ( APPEND PREPRO "JIT_TR2  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__trans_unop" )
        list ( APPEND PREPRO "JIT_TR0  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__union" )
        list ( APPEND PREPRO "JIT_UNI  (" ${F} ")\n" )
    endif ( )
endforeach ( )

list ( JOIN PREPRO "" PREJIT_PROTO )
list ( JOIN PREQUERY "" PREJIT_QUERY )
list ( JOIN PREJIT "\",\n\"" PRENAMES )
list ( LENGTH PREJIT GB_PREJIT_LEN )
list ( JOIN PREJIT ",\n" PREFUNCS )
list ( JOIN PREQ ",\n" PREQFUNCS )


