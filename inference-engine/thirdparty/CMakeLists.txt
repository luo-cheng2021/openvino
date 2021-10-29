# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
function(ie_add_easyjit)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory  ${CMAKE_BINARY_DIR}/easyjit/_install)
    execute_process(COMMAND ${CMAKE_COMMAND} -S${CMAKE_SOURCE_DIR}/inference-engine/thirdparty/easyjit -B${CMAKE_BINARY_DIR}/easyjit -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_COMPILER=/usr/bin/clang-6.0 -DCMAKE_C_COMPILER=/usr/bin/clang-6.0 -DLLVM_DIR=/usr/lib/llvm-6.0/cmake -DEASY_JIT_EXAMPLE=OFF -DEASY_JIT_BENCHMARK=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/easyjit/_install)
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/easyjit --target install)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/easyjit/_install/lib/libEasyJitRuntime.so
        ${CMAKE_BINARY_DIR}/../bin/intel64/${CMAKE_BUILD_TYPE}/lib/)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/easyjit/_install/lib/libEasyJitRuntime.so
        ${CMAKE_INSTALL_PREFIX}/runtime/lib/intel64/)
endfunction()
ie_add_easyjit()
# xsimd
execute_process(COMMAND ${CMAKE_COMMAND} -S${CMAKE_SOURCE_DIR}/inference-engine/thirdparty/xsimd -B${CMAKE_BINARY_DIR}/xsimd -DCMAKE_CXX_FLAGS="-DXSIMD_ENABLE_XTL_COMPLEX=0 " -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/xsimd/_install)
execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/xsimd --target install)

if(ENABLE_MYRIAD)
    add_subdirectory(movidius)
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (MSVC_VERSION VERSION_GREATER_EQUAL "1910"))
    # 1910 version of Visual Studio 2017
    # This flagis needed for enabling SIMD vectorization with command '#pragma omp simd'.
    # Compilation with '/openmp:experimental' key allow us to enable vectorizatikon capability in MSVC.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

if (ENABLE_CLDNN)
    if (ENABLE_TESTS AND ENABLE_CLDNN_TESTS)
        set(CLDNN__INCLUDE_TESTS ON CACHE BOOL "" FORCE)
    else()
        set(CLDNN__INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(clDNN)
endif()

function(ie_add_mkldnn)
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE OFF CACHE BOOL "" FORCE) ## TODO: try it later
    set(DNNL_ENABLE_MAX_CPU_ISA OFF CACHE BOOL "" FORCE)     ## TODO: try it later
    set(DNNL_LIBRARY_TYPE STATIC CACHE BOOL "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_CPU_RUNTIME "${THREADING}" CACHE STRING "" FORCE)
    set(DNNL_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)
    set(DNNL_BLAS_VENDOR "NONE" CACHE STRING "" FORCE)
    set(SDL_cmake_included ON)  ## to skip internal SDL flags. SDL flags are already set on IE level
    if (ANDROID OR ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND NOT (THREADING STREQUAL "OMP")))
        set(OpenMP_cmake_included ON) ## to skip "omp simd" inside a code. Lead to some crashes inside NDK LLVM..
    endif()

    if(SUGGEST_OVERRIDE_SUPPORTED)
        # xbyak compilation fails
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
    endif()
    if(CMAKE_COMPILER_IS_GNUCXX)
        ie_add_compiler_flags(-Wno-undef)
    elseif(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=10121")
    endif()

    add_subdirectory(mkl-dnn EXCLUDE_FROM_ALL)
    add_library(mkldnn ALIAS dnnl)
endfunction()

if(ENABLE_MKL_DNN)
    ie_add_mkldnn()
endif()
