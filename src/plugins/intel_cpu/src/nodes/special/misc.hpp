#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include <thread>
#include <map>
#include <limits>
#include <functional>

//#include "thread_pool.hpp"
#include <openvino/core/type/bfloat16.hpp>


// g++-11 ./test_conv.cpp -O2 -lpthread -march=native && ./a.out

// to use VNNI, we need higher version of compiler:
//    clang-9 ./test_conv.cpp -O2 -lpthread -march=native -lstdc++ && ./a.out

// to use AMX, we need intel compiler 
//   source  ~/intel/oneapi/setvars.sh
//   icx ./mm_amx_bf16.cpp -O2 -lpthread -march=native -lstdc++

// objdump -C -S ./a.out > a.asm

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>


#define rndup(x, n) (((x + n - 1)/n)*n)

template<typename T>
inline void show(const T * data, int rows, int cols) {
    std::ostream& out = std::cout;
    out << "==============\n";
    for(int i0=0; i0 < rows; i0++) {
        out << "[" << i0 << "," << 0 << "]: ";
        for(int i1=0; i1<cols; i1++)
            out << data[i0 * cols + i1] << ",";
        out << std::endl;
    }
}

template<typename T>
inline void vshow(__m512i v) {
    T values[512/8/sizeof(T)];
    _mm512_storeu_si512(values, v);
    show(values, 1, 512/8/sizeof(T));
}

template<typename T>
inline void vshow(__m512 v) {
    T values[512/8/sizeof(T)];
    _mm512_storeu_ps(values, v);
    show(values, 1, 512/8/sizeof(T));
}

struct ANSIcolor {
    const char * code;
    ANSIcolor(const char * code = "0") : code(code){
    }
    friend std::ostream& operator<<(std::ostream& out, const ANSIcolor& obj) {
        out << "\033[" << obj.code << "m";
        return out;
    }
};

