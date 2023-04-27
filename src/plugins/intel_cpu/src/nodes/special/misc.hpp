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
#include <iomanip>
#include <sstream>

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
            //https://stackoverflow.com/questions/14644716/how-to-output-a-character-as-an-integer-through-cout/28414758#28414758
            out << +data[i0 * cols + i1] << ",";
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


struct pretty_size {
    double sz;
    std::string txt;
    pretty_size(double sz, const char * unit = "") : sz(sz) {
        std::stringstream ss;
        ss << std::setprecision(5) << std::setw(5);
        if (sz < 1024)
            ss << sz;
        else if (sz < 1024 * 1024)
            ss << (sz / 1024) << " K";
        else if (sz < 1024 * 1024 * 1024)
            ss << (sz / 1024/1024) << " M";
        else
            ss << (sz / 1024 / 1024/1024) << " G";
        ss << unit;
        txt = ss.str();
    }
    friend std::ostream& operator<<(std::ostream& os, const pretty_size& ps) {
        os << ps.txt;
        return os;
    }
};

inline int readenv(const char * name) {
    int v = 0;
    auto * p = std::getenv(name);
    if (p)
        v = std::atoi(p);
    std::cout << ANSIcolor("32") << "ENV: " << name << " = " << v << std::endl << ANSIcolor();
    return v;
}

//===============================================================
#ifndef _GNU_SOURCE
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#endif
#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

inline bool initXTILE() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    return true;
}
//===============================================================
template<typename T, int tile>
inline void tshow() {
    if (std::is_same<ov::bfloat16,T>::value) {
        ov::bfloat16 data[16*32];
        _tile_stored(tile, data, 64);
        show(data, 16, 32);
    }
    if (std::is_same<float,T>::value) {
        float data[16*16];
        _tile_stored(tile, data, 64);
        show(data, 16, 16);
    }
    if (std::is_same<int8_t,T>::value) {
        int8_t data[16*64];
        _tile_stored(tile, data, 64);
        show(data, 16, 64);
    }
    if (std::is_same<uint8_t,T>::value) {
        uint8_t data[16*64];
        _tile_stored(tile, data, 64);
        show(data, 16, 64);
    }
}

template<typename T0, typename T1>
std::vector<std::pair<T0, T1>> zip_vector(const std::vector<T0> & v0, const std::vector<T1> & v1) {
    std::vector<std::pair<T0, T1>> ret;
    auto sz = v0.size();
    for(decltype(sz) i=0; i<sz; i++)
        ret.push_back(std::make_pair(v0[i], v1[i]));
    return ret;
}

struct tileconfig_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    tileconfig_t() = default;

    tileconfig_t(int palette, int _startRow, const std::vector<std::pair<int, int>> &_rows_columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        int i;
        for(i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for(i = 0; i < _rows_columnsBytes.size(); i++) {
            rows[i] = _rows_columnsBytes[i].first;
            cols[i] = _rows_columnsBytes[i].second;
        }
        for(; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();        
    }

    tileconfig_t(int palette, int _startRow, const std::vector<int> &_rows, int columnsBytes):
        tileconfig_t(palette, _startRow, zip_vector(_rows, std::vector<int>(_rows.size(), columnsBytes))) {
        }
    tileconfig_t(int palette, int _startRow, int numTiles, int _rows, int columnsBytes) :
        tileconfig_t(palette, _startRow, std::vector<std::pair<int, int>>(numTiles, {_rows, columnsBytes})) {
    }

    ~tileconfig_t() {
        _tile_release();
    }
    void __attribute__((noinline)) load() {
        //std::cout << "\ttile load config ... " << std::flush;
        _tile_loadconfig(this);
        //std::cout << *this << std::flush << std::endl;
    }
    void store() {
        _tile_storeconfig(this);
    }
    friend std::ostream& operator<<(std::ostream& out, const tileconfig_t& cfg) {
        out << " palette_id=" << static_cast<int>(cfg.palette_id);
        out << " startRow=" << static_cast<int>(cfg.startRow);
        out << " row x colsb=(";
        for (int i = 0; i < 16;i++) {
            if (cfg.rows[i] == 0 && cfg.cols[i] == 0)
                continue;
            if (i > 0) out << ",";
            out << static_cast<int>(cfg.rows[i]) << "x" << static_cast<int>(cfg.cols[i]);
        }
        out << ")";
        return out;
    }
} __attribute__ ((__packed__));


// default implementation
template <typename T>
struct TypeName {
    static const char* get(){return typeid(T).name();}
};

// a specialization for each type of those you want to support
// and don't like the string returned by typeid
template <>
struct TypeName<int32_t>{
    static const char* get(){return "int32_t";}
};
template <>
struct TypeName<float>{
    static const char* get(){return "foat";}
};
template <>
struct TypeName<ov::bfloat16>{
    static const char* get(){return "bfloat16";}
};
template <>
struct TypeName<int8_t>{
    static const char* get(){return "int8_t";}
};

inline std::ostream & logger() {
    //https://stackoverflow.com/questions/11826554/standard-no-op-output-stream
    static class NullBuffer : public std::streambuf {
    public:
        int overflow(int c) { return c; }
    } null_buffer;
    static std::ostream null_stream(&null_buffer);
    static int log_level = std::getenv("LOGL") ? atoi(std::getenv("LOGL")) : 0;
    return log_level == 0 ? null_stream : std::cout;
}

#ifdef ENABLE_NUMA
static auto USE_NUMA = readenv("USE_NUMA");
#endif
