# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "mkldnn_opt")
project(TARGET_NAME)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/")

file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

if(NOT ENABLE_SS)
    file(GLOB SSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sse42/*.cpp)
    file(GLOB SSE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/sse42/*.h)

    list(APPEND PUBLIC_HEADERS ${SSE_HEADERS})
    list(APPEND LIBRARY_SRC ${SSE_SRC})

    set_source_files_properties(${SSE_SRC} PROPERTIES COMPILE_OPTIONS "-msse4.2")
    add_definitions(-DHAVE_SSE=1)
endif()

if(NOT ENABLE_SS)
    file(GLOB AVX2_SRC ${CMAKE_CURRENT_SOURCE_DIR}/avx2/*.cpp)
    file(GLOB AVX2_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/avx2/*.h)

    list(APPEND PUBLIC_HEADERS ${AVX2_HEADERS})
    list(APPEND LIBRARY_SRC ${AVX2_SRC})

    set_source_files_properties(${AVX2_SRC} PROPERTIES COMPILE_OPTIONS "-mavx2")
    add_definitions(-DHAVE_AVX2=1)
endif()

if(NOT ENABLE_SS)
    file(GLOB AVX512_SRC ${CMAKE_CURRENT_SOURCE_DIR}/avx512/*.cpp)
    file(GLOB AVX512_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/avx512/*.h)

    list(APPEND PUBLIC_HEADERS ${AVX512_HEADERS})
    list(APPEND LIBRARY_SRC ${AVX512_SRC})

    set_source_files_properties(${AVX512_SRC} PROPERTIES COMPILE_OPTIONS "-mavx512f")
    add_definitions(-DHAVE_AVX512=1)
endif()

find_package(EasyJit REQUIRED CONFIG)
include_directories(${EasyJit_INCLUDE_DIRS})
link_directories(${EasyJit_LIBRARY_DIRS})
list(APPEND LINK_LIBRARIES EasyJitRuntime)

# https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Xclang -load -Xclang ${EasyJit_LIBRARY_DIRS}/EasyJitPass.so")

find_package(xsimd REQUIRED CONFIG)
include_directories(${xsimd_INCLUDE_DIRS})

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create library

add_library(${TARGET_NAME}
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

# ie_faster_build(${TARGET_NAME}
#     UNITY
# )

target_link_libraries(${TARGET_NAME} PRIVATE EasyJitRuntime)

target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR})

# LTO

#set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# developer package

#openvino_developer_export_targets(COMPONENT inference_engine TARGETS ${TARGET_NAME})

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION
        LIBRARY DESTINATION)
