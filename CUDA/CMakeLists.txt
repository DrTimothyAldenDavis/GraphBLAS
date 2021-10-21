cmake_minimum_required(VERSION 3.20.1)

# CMake build for generating googletest c++ files that can be compiled and executed in parallel.
# Build can be customized to speed up development by allowing the targeting of specific
# specific parameters. The output of this build is an executable that can be used to
# run the gtests.

project(GRAPHBLAS_CUDA VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "-cudart static")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_STANDARD 99)

message(STATUS "${CMAKE_CXX_FLAGS}")

add_library(graphblascuda SHARED
    GB_AxB_dot3_cuda.cu
#    GB_reduce_to_scalar_cuda.cu
)

set(GRAPHBLAS_CUDA_INCLUDES
        ${RMM_WRAP_INCLUDES}
        ../Source
        ../Source/Template
        ../Include
        ../CUDA)

message(STATUS "${GRAPHBLAS_CUDA_INCLUDES}")


target_include_directories(graphblascuda PUBLIC  ${CUDA_INCLUDE_DIRECTORIES} ${GRAPHBLAS_CUDA_INCLUDES})
set_target_properties(graphblascuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(graphblascuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(graphblascuda PROPERTIES CUDA_ARCHITECTURES "75")

# 1. Execute enumify/stringify/jitify logic to compile ptx kernels and compile/link w/ relevant *.cu files.

# TODO: Need to do this piece in cmake

# 2. Generate test .cu files named "{semiring_operation}_test_instances.hpp"
set(CUDA_TEST_SUITES
    AxB_dot3
#    reduce_to_scalar
)

#
set(CUDA_TEST_MONOIDS PLUS) #MIN MAX TIMES ANY)
set(CUDA_TEST_BINOPS TIMES) #PLUS MIN MAX DIV MINUS RDIV RMINUS FIRST SECOND PAIR)
set(CUDA_TEST_SEMIRINGS PLUS_TIMES) # MIN_PLUS MAX_PLUS)
set(CUDA_TEST_DATATYPES int32_t ) #int64_t uint32_t uint64_t float double)
set(CUDA_TEST_KERNELS mp) # mp vsvs dndn spdn vssp)

# TODO: Update testGen.py to accept the above CUDA_TEST_* params as arguments

# Note: I don't believe there's a way to do this particular piece in parallel but
# once all the files are written, we should be able to compile them in parallel

# Separate individual kernels from larger "overview" test (e.g. 2-level testing structure)
# We want to test all the *_cuda versions

set(CUDA_TEST_CPP_FILES "")
foreach(var ${CUDA_TEST_SUITES})
    foreach(semiring ${CUDA_TEST_SEMIRINGS})
        foreach(kernel ${CUDA_TEST_KERNELS})

            # TODO: Have Python script also build separate cudaTest.cpp (named something
            # like AxB_dot3_cuda_tests.cpp) for each suite. This way we should be able to
            # easily ignore them from the build
            add_custom_command(
                    OUTPUT
                    ${CMAKE_CURRENT_BINARY_DIR}/${var}_${semiring}_${kernel}_test_instances.hpp
                    ${CMAKE_CURRENT_BINARY_DIR}/${var}_${semiring}_${kernel}_cuda_tests.cu
                    DEPENDS
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/jitFactory.hpp
                    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test/testGen_cmake.py "\"${CMAKE_CURRENT_SOURCE_DIR}\"" "\"${var}\"" "\"${CUDA_TEST_MONOIDS}\""
                        "\"${CUDA_TEST_BINOPS}\"" "\"${semiring}\"" "\"${CUDA_TEST_DATATYPES}\""
                        "\"${kernel}\""
            )

            # Construct final list of files to compile (in parallel)
            list(APPEND CUDA_TEST_CPP_FILES ${CMAKE_CURRENT_BINARY_DIR}/${var}_${semiring}_${kernel}_cuda_tests.cu)
        endforeach()
    endforeach()
endforeach()

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_includes)
#execute_process(
#        COMMAND git clone "https://github.com/google/googletest.git" googletest
#        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_includes)
#
#include_directories(${CMAKE_CURRENT_BINARY_DIR}/external_includes/googletest/googletest/include)

#add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/external_includes/googletest/googletest/)

# 3. Compile/link individual {test_suite_name}_cuda_tests.cpp files into a gtest executable
set(GRAPHBLAS_CUDA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/test)

message(STATUS "${CUDA_TEST_CPP_FILES}")

add_executable(graphblascuda_test ${CUDA_TEST_CPP_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/test/run_tests.cpp)
target_include_directories(graphblascuda_test PUBLIC
        ${CUDA_INCLUDE_DIRECTORIES}
        ${GRAPHBLAS_CUDA_INCLUDES})

set_target_properties(graphblascuda_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(graphblascuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(graphblascuda_test PROPERTIES CUDA_ARCHITECTURES "70")

include(GoogleTest)
gtest_discover_tests(graphblascuda_test)

add_dependencies(graphblascuda_test graphblas)
add_dependencies(graphblascuda_test graphblascuda)
add_dependencies(graphblascuda_test gtest_main)

target_link_libraries(graphblascuda_test
        graphblascuda
        graphblas
        gtest_main
        ${ADDITIONAL_DEPS})

target_include_directories(graphblascuda_test PUBLIC
        ${ADDITIONAL_INCLUDES}
        ${CUDA_INCLUDE_DIRECTORIES}
        ${GRAPHBLAS_CUDA_INCLUDES})

add_test(NAME graphblascuda_test_test COMMAND graphblascuda_test)
