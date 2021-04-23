//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "op/batch_norm.hpp"
#include "op/relu.hpp"
#include "op/pool2d.hpp"
#include "op/elementwise_ops.hpp"
#include "op/conv2d.hpp"
#include "op/matmul.hpp"
#include "op/mul.hpp"
#include "op/pool2d.hpp"
#include "op/relu.hpp"
#include "op/reshape2.hpp"
#include "op/scale.hpp"
#include "op/leakyrelu.hpp"
#include "op/interp.hpp"
#include "op/concat.hpp"
#include "op/cast.hpp"
#include "op/softmax.hpp"
#include "op/split.h"
#include "op/transpose2.hpp"
#include "op/yolo_box.hpp"
#include "op/multiclass_nms.hpp"
#include "op/rnn.hpp"
#include "op/fill_constant.hpp"
#include "op/uniform_random.hpp"
#include "op/assign_value.hpp"
#include "op/sigmoid.hpp"
#include "op/hard_sigmoid.hpp"
#include "op/relu6.hpp"
#include "op/pow.hpp"
#include "op/squeeze.hpp"
#include "op/unsqueeze.hpp"
#include "op/slice.hpp"
#include "op/hard_swish.hpp"

#include "op_table.hpp"


namespace ngraph {
namespace frontend {
namespace pdpd {

std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
            {"batch_norm", op::batch_norm},
            {"conv2d", op::conv2d},
            {"elementwise_add", op::elementwise_add},
            {"elementwise_sub", op::elementwise_sub},
            {"elementwise_mul", op::elementwise_mul},
            {"elementwise_div", op::elementwise_div},
            {"elementwise_min", op::elementwise_min},
            {"elementwise_max", op::elementwise_max},
            {"elementwise_pow", op::elementwise_pow},
            {"matmul", op::matmul},
            {"mul", op::mul},
            {"pool2d", op::pool2d},
            {"relu", op::relu},
            {"reshape2", op::reshape2},
            {"scale", op::scale},
            {"leaky_relu", op::leaky_relu},
            {"nearest_interp_v2", op::nearest_interp_v2},
            {"bilinear_interp_v2", op::bilinear_interp_v2},
            {"nearest_interp", op::nearest_interp_v2},
            {"bilinear_interp", op::bilinear_interp_v2},
            {"concat", op::concat},
            {"cast", op::cast},
            {"split", op::split},
            {"max_pool2d_with_index", op::pool2d}, //adaptive_max_pool2d
            {"softmax", op::softmax},
            {"transpose2", op::transpose2},
            {"yolo_box", op::yolo_box},
            {"multiclass_nms3", op::multiclass_nms},
            {"rnn", op::rnn},
            {"fill_constant", op::fill_constant},
            {"bmm", op::matmul},
            {"depthwise_conv2d", op::conv2d},
            {"uniform_random", op::uniform_random},
            {"assign_value", op::assign_value},
            {"sigmoid", op::sigmoid},
            {"hard_sigmoid", op::hard_sigmoid},
            {"relu6", op::relu6},
            {"pow", op::pow},
            {"squeeze2", op::squeeze},
            {"unsqueeze2", op::unsqueeze},
            {"slice", op::slice},
            {"hard_swish", op::hard_swish}
        };
};

}}}
