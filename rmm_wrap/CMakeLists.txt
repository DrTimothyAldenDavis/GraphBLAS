cmake_minimum_required(VERSION 3.20.1)

project(RMM_Wrap VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

find_package(rmm 21.8)

configure_file(RMM_WrapConfig.h.in RMM_WrapConfig.h)

add_executable(RMM_Wrap_Test rmm_wrap_test.c)

list(APPEND EXTRA_INCLUDES $CONDA_PREFIX/include /usr/local/cuda)

target_link_libraries(RMM_Wrap_Test PUBLIC  
	              ${EXTRA_LIBS}
					Thrust
		      		rmm::rmm)

target_include_directories(RMM_Wrap_Test PUBLIC
		"${PROJECT_BINARY_DIR}"
		"${EXTRA_INCLUDES}"
		"${PROJECT_SOURCE_DIR}/include")
