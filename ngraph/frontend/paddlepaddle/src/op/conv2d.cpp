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

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/builder/reshape.hpp>
#include "conv2d.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

ngraph::op::PadType get_auto_pad(const NodeContext& node)
{
    // Default value means use explicitly provided padding values.
    ngraph::op::PadType pad_type{ngraph::op::PadType::NOTSET};
    auto padding_algorithm = node.get_attribute<std::string>("padding_algorithm");
    static std::unordered_multimap<std::string, ngraph::op::PadType>
            auto_pad_values{
            {"VALID", ngraph::op::PadType::VALID},
            {"SAME", ngraph::op::PadType::SAME_UPPER},
            {"NOTSET", ngraph::op::PadType::NOTSET},
    };

    const auto pad_val_it = auto_pad_values.find(padding_algorithm);

    if(pad_val_it == auto_pad_values.end()) {
        pad_type = ngraph::op::PadType::NOTSET;
    } else {
        pad_type = pad_val_it->second;
    }



    return pad_type;
}

std::pair<CoordinateDiff, CoordinateDiff> get_pads(const NodeContext& node,
                                                   const size_t kernel_rank)
{
    CoordinateDiff pads(kernel_rank, 0);

    auto pads_int32 = node.get_attribute<std::vector<int32_t>>("paddings");
    pads = CoordinateDiff{std::begin(pads_int32), std::end(pads_int32)};
    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;


    if (pads.size() == kernel_rank * 2)
    {
        for(int i = 0; i < pads.size(); i++)
        {
            if(i & 0x01)
            {
                pads_end.push_back(pads[i]);
            } else {
                pads_begin.push_back(pads[i]);
            }
        }
        return {pads_begin, pads_end};
    }
    else
    {
        // No paddings provided or only one side values provided, which means same
        // padding at both begin and end of axis.
        return {pads, pads};
    }
}

std::pair<CoordinateDiff, CoordinateDiff> get_pads(const NodeContext& node)
{
    const auto data_rank = node.get_ng_input("Input").get_partial_shape().rank();

    const auto data_spatial_dims = data_rank.get_length() - 2;

    return get_pads(node, data_spatial_dims);
}

OutputVector conv2d (const NodeContext& node) {
    auto data = node.get_ng_input("Input");
    auto filters = node.get_ng_input("Filter");
    // TODO: resolve padding according to spec
    const auto strides = node.get_attribute<std::vector<int32_t>>("strides");
    const auto dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    const auto auto_pad_type = get_auto_pad(node);
    const auto paddings = get_pads(node);
    const auto pads_begin = paddings.first;
    const auto pads_end = paddings.second;
    const auto groups = node.get_attribute<int32_t>("groups");

    if (groups > 1) {
        auto filters_shape = filters.get_shape();
        filters_shape.at(0) = filters_shape.at(0) / groups;
        filters_shape.insert(filters_shape.begin(), groups);

        const auto reshaped_filters =
                ngraph::builder::opset1::reshape(filters, filters_shape);
        return {std::make_shared<opset6::GroupConvolution>(
                data,
                reshaped_filters,
                ngraph::Strides(strides.begin(), strides.end()),
                pads_begin,
                pads_end,
                ngraph::Strides(dilations.begin(), dilations.end()),
                auto_pad_type)};
    }
    else
    {
        return {std::make_shared<opset6::Convolution>(
                    data,
                    filters,
                    ngraph::Strides(strides.begin(), strides.end()),
                    pads_begin,
                    pads_end,
                    ngraph::Strides(dilations.begin(), dilations.end()),
                    auto_pad_type)};
    }
}

}}}}