// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// This file include all fused algorithm function type declare. Compiling in both gcc and clang.
// Note: FuseConstParams will be inlined in the jit code and do not use pointer inside it
#include <memory.h>
#include <stdint.h>

enum class AlgType {
    // Unary: x = f(x)
    Abs,
    // Binary: x = f(x, y), x and y are variable
    Add,
    Sub,
    Mul,
    // BinaryConst: x = f(x, c1, c2), x is varible and c1/c2 is const
    Add_C,
    Sub_C,
    Mul_C,
};

// const fuse parameter at runtime
struct FuseConstAlgParam {
    //int type;           // paramter types: 0 float, 1 int8, 2 uint8 etc. TODO: use standard types
    // Binary const will use
    float x1;
    float x2;
    float x3;
    float x4;
};

// mutable fuse parameter at runtime
struct FuseMutableAlgParam {
    // Binary will use
    uint64_t addr;         // address
    int stride_x;          // x stride, unit is bytes
    int stride_xy;         // x*y offset, unit is bytes
};
#define MAX_FUSE_NUM 10
// const fuse algorithm and its parameter at runtime
struct FuseConstParams {
    AlgType types[MAX_FUSE_NUM];
    int num;
    FuseConstAlgParam params[MAX_FUSE_NUM];
    FuseConstParams() {
        memset(this, 0, sizeof(*this));
    }
};

// mutable fuse parameter at runtime
struct FuseMutableParams {
    FuseMutableAlgParam params[MAX_FUSE_NUM];   // the num should equal FuseConstParams.num
    FuseMutableParams() {
        memset(this, 0, sizeof(*this));
    }
};
