# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:functional_tests]
set(TARGET_NAME templateFuncTests)

find_package(EasyJit REQUIRED CONFIG
    PATHS ${CMAKE_BINARY_DIR}/easyjit/_install/lib/cmake
    NO_DEFAULT_PATH)
include_directories(${EasyJit_INCLUDE_DIRS})
link_directories(${EasyJit_LIBRARY_DIRS})

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            templatePlugin
        LINK_LIBRARIES
            IE::funcSharedTests
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}mkldnn_opt${CMAKE_STATIC_LIBRARY_SUFFIX}
            EasyJitRuntime
        INCLUDES
            "${IE_MAIN_TEMPLATE_PLUGIN_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/op_reference"
        ADD_CPPLINT
        LABELS
            TEMPLATE
)

if(ENABLE_HETERO)
    add_dependencies(${TARGET_NAME} HeteroPlugin)
endif()

find_package(OpenCV QUIET COMPONENTS core imgproc)

if(OpenCV_FOUND)
    message("-- Reference preprocessing: OpenCV tests are enabled")
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_TEMPLATE_TESTS)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_imgproc opencv_core)
else()
    message("-- Reference preprocessing: OpenCV tests are disabled")
endif()

# [cmake:functional_tests]
