#pragma once

#include <stdint.h>
#include <openvino/core/type/bfloat16.hpp>

namespace ov {
namespace intel_cpu {

void mvn_line(bfloat16* src, size_t ele_num, float eps, bool inside_sqrt, bfloat16 *dst);
void mvn_line(bfloat16* src, size_t ele_num, float eps, bool inside_sqrt, int8_t *dst, float* quant);

}
}