#pragma once

#include <stdint.h>
#include <openvino/core/type/bfloat16.hpp>

namespace ov {
namespace intel_cpu {

void add3(bfloat16* a, bfloat16 *b, bfloat16 *c, bfloat16 *dst, size_t ele_num);

}
}