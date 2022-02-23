# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "opt_test")
project(TARGET_NAME)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/")

file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

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
