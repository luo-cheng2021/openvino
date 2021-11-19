# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "MKLDNNPlugin")

find_package(EasyJit REQUIRED CONFIG
    PATHS ${CMAKE_BINARY_DIR}/easyjit/_install/lib/cmake
    NO_DEFAULT_PATH)
include_directories(${EasyJit_INCLUDE_DIRS})
link_directories(${EasyJit_LIBRARY_DIRS})
list(APPEND LINK_LIBRARIES EasyJitRuntime)
# change cc to clang
include(ExternalProject)
set(MKLDNN_OPT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkldnn_opt/")
set(MKLDNN_OPT_INSTALL_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/")
set(MKLDNN_OPT_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/mkldnn_opt/${CMAKE_STATIC_LIBRARY_PREFIX}mkldnn_opt${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(MKLDNN_OPT_OUTPUT_LIBRARY "${MKLDNN_OPT_INSTALL_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkldnn_opt${CMAKE_STATIC_LIBRARY_SUFFIX}")

ExternalProject_Add(mkldnn_opt
PREFIX ${NAME}
SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nodes/opt"
BINARY_DIR "${MKLDNN_OPT_BUILD_DIR}"
INSTALL_DIR "${MKLDNN_OPT_INSTALL_DIR}"
PREFIX mkldnn_opt
CMAKE_ARGS "-DCMAKE_CXX_COMPILER=clang-6.0"
           "-DCMAKE_C_COMPILER=clang-6.0"
           "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
           "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
           "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
           "-Dxsimd_DIR=${CMAKE_BINARY_DIR}/xsimd/_install/lib/cmake/xsimd"
           "-DEasyJit_DIR=${CMAKE_BINARY_DIR}/easyjit/_install/lib/cmake"
BUILD_BYPRODUCTS ${MKLDNN_OPT_OUTPUT_LIBRARY}
BUILD_ALWAYS TRUE
BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target mkldnn_opt --parallel
INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_OPT_LIBRARY} ${MKLDNN_OPT_OUTPUT_LIBRARY}
COMMAND ${CMAKE_COMMAND} -E echo "mkldnn opt build complete")

if(CMAKE_COMPILER_IS_GNUCC)
    ie_add_compiler_flags(-Wno-all)
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

if (ENABLE_CPU_DEBUG_CAPS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_DEBUG_CAPS")
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

set(SOURCES)
set(HEADERS)
# Store unfiltered list of files into temporary list
file(GLOB_RECURSE _tmp_list ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# Resulted list is initially empty. Copy all needed elements into it.
foreach(f ${_tmp_list})
    if(NOT f MATCHES "/opt/")
        list(APPEND SOURCES ${f})
    endif()
endforeach()

file(GLOB_RECURSE _tmp_list ${CMAKE_CURRENT_SOURCE_DIR}/*.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
foreach(f ${_tmp_list})
    if(NOT f MATCHES "/opt/")
        list(APPEND HEADERS ${f})
    endif()
endforeach()


addVersionDefines(mkldnn_plugin.cpp CI_BUILD_NUMBER)

# create plugin

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              AS_EXTENSION
              SOURCES ${SOURCES} ${HEADERS})

set_ie_threading_interface_for(${TARGET_NAME})

ie_mark_target_as_cc(${TARGET_NAME})

if(SELECTIVE_BUILD STREQUAL "ON")
    # After disabling a block of code, some variables might be unused.
    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        target_compile_options(${TARGET_NAME} PRIVATE -Wno-unused-variable)
    endif()
endif()

add_library(mkldnn_opt_lib STATIC IMPORTED)
set_property(TARGET mkldnn_opt_lib PROPERTY
    IMPORTED_LOCATION ${MKLDNN_OPT_OUTPUT_LIBRARY})
add_dependencies(${TARGET_NAME} mkldnn_opt)

target_link_libraries(${TARGET_NAME} PRIVATE mkldnn
                                             inference_engine
                                             inference_engine_transformations
                                             inference_engine_lp_transformations
                                             ov_shape_inference
                                             mkldnn_opt_lib
                                             EasyJitRuntime)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:mkldnn,INCLUDE_DIRECTORIES>)

# Cross compiled function
# TODO: The same for proposal, proposalONNX, topk
cross_compiled_file(${TARGET_NAME}
        ARCH AVX2 ANY
                    nodes/proposal_imp.cpp
        API         nodes/proposal_imp.hpp
        NAME        proposal_exec
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

#  add test object library

add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
link_system_libraries(${TARGET_NAME}_obj PUBLIC mkldnn PRIVATE EasyJitRuntime ${MKLDNN_OPT_OUTPUT_LIBRARY})

target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:ov_shape_inference,INTERFACE_INCLUDE_DIRECTORIES>
                                              PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
                                                      $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:mkldnn,INCLUDE_DIRECTORIES>)

set_ie_threading_interface_for(${TARGET_NAME}_obj)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE
    USE_STATIC_IE IMPLEMENT_INFERENCE_ENGINE_PLUGIN IMPLEMENT_INFERENCE_EXTENSION_API
    $<TARGET_PROPERTY:ngraph,INTERFACE_COMPILE_DEFINITIONS>
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_COMPILE_DEFINITIONS>
)

set_target_properties(${TARGET_NAME}_obj PROPERTIES EXCLUDE_FROM_ALL ON)

# LTO

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_obj
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
