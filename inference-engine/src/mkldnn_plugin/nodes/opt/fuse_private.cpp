// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This file includes all fused function needed by the optimization function
// Note: 1, Common function should be static to avoid linker pick the wrong architecture version
// 2, if want use specific instrisc can use the predefined macro: HAVE_AVX2/HAVE_SSE42/HAVE_AVX512

#include <utility>
#include <map>
#include "fuse_private.h"
#include "xsimd.h"
