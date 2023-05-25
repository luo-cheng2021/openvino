// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/op/op.hpp"
#include "mvn.hpp"

namespace ov {
namespace op {

namespace v10 {
/// \brief Operator performing Mean Variance Normalization
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API MVNCustom : public Op {
public:
    OPENVINO_OP("MVNCustom", "opset10");

    MVNCustom() = default;
    /// \brief Constructs an MVNCustom operation.
    ///
    /// \param data Input tensor with data
    /// \param reduction_axes A list of axes, along which to reduce.
    /// \param normalize_variance flag that denotes whether to perform variance
    ///                           normalization.
    /// \param eps the number to be added to the variance to avoid division by zero when
    ///            normalizing the value
    /// \param eps_mode the mode of applying epsilon
    ///
    MVNCustom(const Output<Node>& data,
        const Output<Node>& reduction_axes,
        const Output<Node>& weight,
        const Output<Node>& bias,
        bool normalize_variance,
        float eps,
        MVNEpsMode eps_mode);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    float get_eps() const {
        return m_eps;
    }
    void set_eps(const float& eps) {
        m_eps = eps;
    }
    bool get_normalize_variance() const {
        return m_normalize_variance;
    }
    void set_normalize_variance(bool normalize_variance) {
        m_normalize_variance = normalize_variance;
    }
    MVNEpsMode get_eps_mode() const {
        return m_eps_mode;
    }
    void set_eps_mode(const MVNEpsMode& eps_mode) {
        m_eps_mode = eps_mode;
    }

private:
    bool m_normalize_variance;
    float m_eps;
    MVNEpsMode m_eps_mode;
};
}  // namespace v6
}  // namespace op

}  // namespace ov
