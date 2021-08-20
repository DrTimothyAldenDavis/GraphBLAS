cmake_minimum_required(VERSION 3.20.1)

# CMake build for generating googletest c++ files that can be compiled and executed in parallel.
# Build can be customized to speed up development by allowing the targeting of specific
# specific parameters. The output of this build is an executable that can be used to
# run the gtests.

project(GRAPHBLAS_CUDA VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)

# The ou

# 1. Execute enumify/stringify/jitify logic to compile ptx kernels and compile/link w/ relevant *.cu files.

# TODO: Need to do this piece in cmake

# 2. Generate test .cu files named "{semiring_operation}_test_instances.hpp"
list(CUDA_TEST_SUITES
    AxB_dot3
)


list(CUDA_TEST_MONOIDS PLUS MIN MAX TIMES ANY)
list(CUDA_TEST_BINOPS TIMES PLUS MIN MAX DIV MINUS RDIV RMINUS FIRST SECOND PAIR)
list(CUDA_TEST_SEMIRINGS PLUS_TIMES MIN_PLUS MAX_PLUS)
list(CUDA_TEST_DATATYPES int32_t int64_t uint32_t uint64_t float double)
list(CUDA_TEST_KERNELS warp mp vsvs dndn spdn vssp)

# TODO: Update testGen.py to accept the above CUDA_TEST_* params as arguments

# Note: I don't believe there's a way to do this particular piece in parallel but
# once all the files are written, we should be able to compile them in parallel

list(CUDA_TEST_CPP_FILES)
foreach(var ${CUDA_TEST_SUITES})

    # TODO: Have Python script also build separate cudaTest.cpp (named something
    # like AxB_dot3_cuda_tests.cpp) for each suite. This way we should be able to
    # easily ignore them from the build
    add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_SOURCE_DIR}/${var}_test_instances.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/${var}_cuda_tests.cpp
            DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/jitFactory.hpp
            COMMAND python testGen_cmake.py "${var}" "${CUDA_TEST_MONOIDS}"
                "${CUDA_TEST_BINOPS}" "${CUDA_TEST_SEMIRINGS}" "${CUDA_TEST_DATATYPES}"
                "${CUDA_TEST_KERNELS}"
    )

    # Construct final list of files to compile (in parallel)
    list(APPEND CUDA_TEST_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${var}_cuda_tests.cpp)
endforeach()

# 3. Compile/link individual {test_suite_name}_cuda_tests.cpp files into a gtest executable

add_executable(${CUDA_TEST_TARGET} ${CUDA_TEST_CPP_FILES})
add_target_link_libraries(${CUDA_TEST_TARGET} "graphblas" "graphblascuda.so" ${ADDITIONAL_DEPS})
add_target_include_libraries(${CUDA_TEST_TARGET} ${ADDITIONAL_INCLUDES})
