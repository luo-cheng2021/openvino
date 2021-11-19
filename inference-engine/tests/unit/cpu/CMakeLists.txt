# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuUnitTests)

find_package(EasyJit REQUIRED CONFIG
    PATHS ${CMAKE_BINARY_DIR}/easyjit/_install/lib/cmake
    NO_DEFAULT_PATH)
include_directories(${EasyJit_INCLUDE_DIRS})
link_directories(${EasyJit_LIBRARY_DIRS})

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            $<TARGET_PROPERTY:MKLDNNPlugin,SOURCE_DIR>
            $<TARGET_PROPERTY:MKLDNNPlugin,SOURCE_DIR>/nodes
            $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>
        OBJECT_FILES
            $<TARGET_OBJECTS:MKLDNNPlugin_obj>
        LINK_LIBRARIES
            gtest
            gtest_main
            gmock
            mkldnn
            inference_engine_transformations
            inference_engine_lp_transformations
            ov_shape_inference
            inference_engine_s
            unitTestUtils
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}mkldnn_opt${CMAKE_STATIC_LIBRARY_SUFFIX}
            EasyJitRuntime
        ADD_CPPLINT
        LABELS
            CPU
)

ie_faster_build(${TARGET_NAME}
    UNITY
)
