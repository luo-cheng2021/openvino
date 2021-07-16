// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_matrix_nms_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "base.hpp"
#include "ie_parallel.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph_ops/nms_static_shape_ie.hpp"
#include "utils/general_utils.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using MatrixNmsIEInternal = ngraph::op::internal::NmsStaticShapeIE<ngraph::op::v8::MatrixNms>;

using ngNmsSortResultType = ngraph::op::util::NmsBase::SortResultType;
using ngNmseDcayFunction = ngraph::op::v8::MatrixNms::DecayFunction;

bool MKLDNNMatrixNmsNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto nms = std::dynamic_pointer_cast<const MatrixNmsIEInternal>(op);
        if (!nms) {
            errorMessage = "Only internal MatrixNms operation is supported";
            return false;
        }
        const auto& attrs = nms->get_attrs();
        const auto& sortType = attrs.sort_result_type;
        if (!one_of(sortType, ngNmsSortResultType::NONE, ngNmsSortResultType::SCORE, ngNmsSortResultType::CLASSID)) {
            errorMessage = "Doest not support SortResultType";
            return false;
        }
        const auto& decayType = attrs.decay_function;
        if (!one_of(decayType, ngNmseDcayFunction::LINEAR, ngNmseDcayFunction::GAUSSIAN)) {
            errorMessage = "Does not support DcayFunction";
            return false;
        }

        if (nms->get_input_shape(NMS_BOXES)[1] != nms->get_input_shape(NMS_SCORES)[2]) {
            errorMessage = "Input Box Dimension 1: " + std::to_string(nms->get_input_shape(NMS_BOXES)[1]) +
                           " doesn't match Score Dimension 2: " + std::to_string(nms->get_input_shape(NMS_SCORES)[2]);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMatrixNmsNode::MKLDNNMatrixNmsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr& cache)
    : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "MatrixNMS layer with name '" + getName() + "' ";
    const auto matrix_nms = std::dynamic_pointer_cast<const MatrixNmsIEInternal>(op);

    if (getOriginalInputsNumber() != 2)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << matrix_nms->get_input_size();

    if (getOriginalOutputsNumber() < 1 || matrix_nms->get_output_size() > 3)
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << matrix_nms->get_output_size();

    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32, Precision::BF16};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inType);

    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTED_OUTPUTS), supportedFloatPrecision, "selected_outputs", outType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTED_INDICES), supportedIntOutputPrecision, "selected_indices", outType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_VALID_OUTPUTS), supportedIntOutputPrecision, "valid_outputs", outType);

    outputShape_SELECTED_OUTPUTS = op->get_output_shape(NMS_SELECTED_OUTPUTS);
    outputShape_SELECTED_INDICES = op->get_output_shape(NMS_SELECTED_INDICES);
    outputShape_VALID_OUTPUTS = op->get_output_shape(NMS_VALID_OUTPUTS);

    const SizeVector& boxes_dims = op->get_input_shape(NMS_BOXES);
    m_numBatches = boxes_dims[0];
    m_numBoxes = boxes_dims[1];
    if (boxes_dims.size() != 3)
        IE_THROW() << errorPrefix << "has unsupported 'boxes' input rank: " << boxes_dims.size();
    if (boxes_dims[2] != 4)
        IE_THROW() << errorPrefix << "has unsupported 'boxes' input 3rd dimension size: " << boxes_dims[2];
    const SizeVector& scores_dims = op->get_input_shape(NMS_SCORES);
    m_numClasses = scores_dims[1];
    if (scores_dims.size() != 3)
        IE_THROW() << errorPrefix << "has unsupported 'scores' input rank: " << scores_dims.size();

    if (m_numBatches != scores_dims[0])
        IE_THROW() << errorPrefix << " num_batches is different in 'boxes' and 'scores' inputs";
    if (m_numBoxes != scores_dims[2])
        IE_THROW() << errorPrefix << " num_boxes is different in 'boxes' and 'scores' inputs";
    auto& attrs = matrix_nms->get_attrs();
    if (attrs.sort_result_type == ngraph::op::util::NmsBase::SortResultType::CLASSID)
        m_sortResultType = MatrixNmsSortResultType::CLASSID;
    else if (attrs.sort_result_type == ngraph::op::util::NmsBase::SortResultType::SCORE)
        m_sortResultType = MatrixNmsSortResultType::SCORE;
    else if (attrs.sort_result_type == ngraph::op::util::NmsBase::SortResultType::NONE)
        m_sortResultType = MatrixNmsSortResultType::NONE;

    if (attrs.decay_function == ngraph::op::v8::MatrixNms::DecayFunction::GAUSSIAN)
        m_decayFunction = GAUSSIAN;
    else if (attrs.decay_function == ngraph::op::v8::MatrixNms::DecayFunction::LINEAR)
        m_decayFunction = LINEAR;

    m_sortResultAcrossBatch = attrs.sort_result_across_batch;
    m_outputType = attrs.output_type;
    m_scoreThreshold = attrs.score_threshold;
    m_nmsTopk = attrs.nms_top_k;
    m_keepTopk = attrs.keep_top_k;
    m_backgroundClass = attrs.background_class;

    m_gaussianSigma = attrs.gaussian_sigma;
    m_postThreshold = attrs.post_threshold;
    m_normalized = attrs.normalized;
    int64_t max_output_boxes_per_class = 0;
    size_t real_num_classes = m_backgroundClass == -1 ? m_numClasses : m_numClasses - 1;
    if (m_nmsTopk >= 0)
        max_output_boxes_per_class = std::min(m_numBoxes, static_cast<size_t>(m_nmsTopk));
    else
        max_output_boxes_per_class = m_numBoxes;

    m_maxBoxesPerBatch = max_output_boxes_per_class * real_num_classes;
    if (m_keepTopk >= 0)
        m_maxBoxesPerBatch = std::min(m_maxBoxesPerBatch, static_cast<size_t>(m_keepTopk));
}

void MKLDNNMatrixNmsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    m_realNumClasses = m_backgroundClass == -1 ? m_numClasses : m_numClasses - 1;
    m_realNumBoxes = m_nmsTopk == -1 ? m_numBoxes : std::min(m_nmsTopk, static_cast<int>(m_numBoxes));
    m_numPerBatch.resize(m_numBatches);
    m_filteredBoxes.resize(m_numBatches * m_realNumClasses * m_realNumBoxes);

    if (m_decayFunction == MatrixNmsDecayFunction::LINEAR) {
        m_decay_fn = [](float iou, float max_iou, float sigma) -> float {
          return (1. - iou) / (1. - max_iou + 1e-10f);
        };
    } else {
        m_decay_fn = [](float iou, float max_iou, float sigma) -> float {
          return std::exp((max_iou * max_iou - iou * iou) * sigma);
        };
    }


    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inType);
    checkOutput(outputShape_VALID_OUTPUTS, supportedIntOutputPrecision, "valid_outputs", NMS_VALID_OUTPUTS);
    checkOutput(outputShape_SELECTED_OUTPUTS, supportedFloatPrecision, "selected_outputs", NMS_SELECTED_OUTPUTS);
    checkOutput(outputShape_SELECTED_INDICES, supportedIntOutputPrecision, "selected_indices", NMS_SELECTED_INDICES);

    std::vector<DataConfigurator> inDataConf(NMS_SCORES + 1, {TensorDescCreatorTypes::ncsp, Precision::FP32});

    std::vector<DataConfigurator> outDataConf;
    outDataConf.reserve(getOriginalOutputsNumber());
    for (int i = 0; i < getOriginalOutputsNumber(); ++i) {
        Precision outPrecision = i == NMS_SELECTED_OUTPUTS ? Precision::FP32 : Precision::I32;
        outDataConf.emplace_back(TensorDescCreatorTypes::ncsp, outPrecision);
    }

    addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
}

bool MKLDNNMatrixNmsNode::created() const {
    return getType() == MatrixNms;
}

namespace {

static inline float boxArea(const float* box, const bool normalized) {
    if (box[2] < box[0] || box[3] < box[1]) {
        // If coordinate values are is invalid
        // (e.g. xmax < xmin or ymax < ymin), return 0.
        return static_cast<float>(0.);
    } else {
        const float w = box[2] - box[0];
        const float h = box[3] - box[1];
        if (normalized) {
            return w * h;
        } else {
            // If coordinate values are not within range [0, 1].
            return (w + 1) * (h + 1);
        }
    }
}

static inline float intersectionOverUnion(const float* box1, const float* box2, const bool normalized) {
    if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] || box2[3] < box1[1]) {
        return static_cast<float>(0.);
    } else {
        const float inter_xmin = std::max(box1[0], box2[0]);
        const float inter_ymin = std::max(box1[1], box2[1]);
        const float inter_xmax = std::min(box1[2], box2[2]);
        const float inter_ymax = std::min(box1[3], box2[3]);
        float norm = normalized ? static_cast<float>(0.) : static_cast<float>(1.);
        float inter_w = inter_xmax - inter_xmin + norm;
        float inter_h = inter_ymax - inter_ymin + norm;
        const float inter_area = inter_w * inter_h;
        const float bbox1_area = boxArea(box1, normalized);
        const float bbox2_area = boxArea(box2, normalized);
        return inter_area / (bbox1_area + bbox2_area - inter_area);
    }
}
}  // namespace

size_t MKLDNNMatrixNmsNode::nmsMatrix(const float* boxes_data, const float* scores_data, BoxInfo* filterBoxes,
                                      const int64_t batchIdx, const int64_t classIdx) {
    std::vector<int32_t> candidate_index(m_numBoxes);
    std::iota(candidate_index.begin(), candidate_index.end(), 0);
    auto end = std::remove_if(candidate_index.begin(), candidate_index.end(), [&scores_data, this](int32_t idx) {
        return scores_data[idx] <= m_scoreThreshold;
    });
    int64_t num_det = 0;
    int64_t original_size = std::distance(candidate_index.begin(), end);
    if (original_size <= 0) {
        return 0;
    }
    if (m_nmsTopk > -1 && original_size > m_nmsTopk) {
        original_size = m_nmsTopk;
    }

    std::partial_sort(candidate_index.begin(), candidate_index.begin() + original_size, end, [&scores_data](int32_t a, int32_t b) {
        return scores_data[a] > scores_data[b];
    });

    std::vector<float> iou_matrix((original_size * (original_size - 1)) >> 1);
    std::vector<float> iou_max(original_size);

    iou_max[0] = 0.;
    InferenceEngine::parallel_for(original_size - 1, [&](size_t i) {
        float max_iou = 0.;
        size_t actual_index = i + 1;
        auto idx_a = candidate_index[actual_index];
        for (int64_t j = 0; j < actual_index; j++) {
            auto idx_b = candidate_index[j];
            auto iou = intersectionOverUnion(boxes_data + idx_a * 4, boxes_data + idx_b * 4, m_normalized);
            max_iou = std::max(max_iou, iou);
            iou_matrix[actual_index * (actual_index - 1) / 2 + j] = iou;
        }
        iou_max[actual_index] = max_iou;
    });

    if (scores_data[candidate_index[0]] > m_postThreshold) {
        auto box_index = candidate_index[0];
        auto box = boxes_data + box_index * 4;
        filterBoxes[0].box.x1 = box[0];
        filterBoxes[0].box.y1 = box[1];
        filterBoxes[0].box.x2 = box[2];
        filterBoxes[0].box.y2 = box[3];
        filterBoxes[0].index = batchIdx * m_numBoxes + box_index;
        filterBoxes[0].score = scores_data[candidate_index[0]];
        filterBoxes[0].batch_index = batchIdx;
        filterBoxes[0].class_index = classIdx;
        num_det++;
    }

    for (int64_t i = 1; i < original_size; i++) {
        float min_decay = 1.;
        for (int64_t j = 0; j < i; j++) {
            auto max_iou = iou_max[j];
            auto iou = iou_matrix[i * (i - 1) / 2 + j];
            auto decay = m_decay_fn(iou, max_iou, m_gaussianSigma);
            min_decay = std::min(min_decay, decay);
        }
        auto ds = min_decay * scores_data[candidate_index[i]];
        if (ds <= m_postThreshold)
            continue;
        auto box_index = candidate_index[i];
        auto box = boxes_data + box_index * 4;
        filterBoxes[num_det].box.x1 = box[0];
        filterBoxes[num_det].box.y1 = box[1];
        filterBoxes[num_det].box.x2 = box[2];
        filterBoxes[num_det].box.y2 = box[3];
        filterBoxes[num_det].index = batchIdx * m_numBoxes + box_index;
        filterBoxes[num_det].score = ds;
        filterBoxes[num_det].batch_index = batchIdx;
        filterBoxes[num_det].class_index = classIdx;
        num_det++;
    }
    return num_det;
}

void MKLDNNMatrixNmsNode::execute(mkldnn::stream strm) {
    const float* boxes = reinterpret_cast<const float*>(getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->GetPtr());
    const float* scores = reinterpret_cast<const float*>(getParentEdgeAt(NMS_SCORES)->getMemoryPtr()->GetPtr());
    InferenceEngine::parallel_for(m_numBatches, [&](size_t batch) {
        const float* boxes_ptr = boxes + batch * m_numBoxes * 4;
        std::vector<BoxInfo> batchFilteredBox(m_realNumClasses * m_realNumBoxes);
        std::vector<int> class_offset(m_numClasses, 0);
        std::vector<int64_t> num_per_class(m_numClasses, 0);
        for (size_t i = 0, count = 0; i < m_numClasses; i++) {
            if (i == m_backgroundClass)
                continue;
            class_offset[i] = (count++) * m_realNumBoxes;
        }

        int64_t num_det = 0;
        InferenceEngine::parallel_for(m_numClasses, [&](size_t class_idx) {
            if (class_idx == m_backgroundClass)
                return;
            const float* scores_ptr = scores + batch * (m_numClasses * m_numBoxes) + class_idx * m_numBoxes;
            size_t classNumDet = 0;
            classNumDet = nmsMatrix(boxes_ptr, scores_ptr, batchFilteredBox.data() + class_offset[class_idx], batch, class_idx);
            num_per_class[class_idx] = classNumDet;
        });
        num_det = std::accumulate(num_per_class.begin(), num_per_class.end(), 0);
        if (num_det <= 0) {
            return;
        }

        auto start_offset = num_per_class[0];
        for (size_t i = 1; i < num_per_class.size(); i++) {
            auto offset_class = class_offset[i];
            for (size_t j = 0; j < num_per_class[i]; j++) {
                batchFilteredBox[start_offset + j] = batchFilteredBox[offset_class + j];
            }
            start_offset += num_per_class[i];
        }

        batchFilteredBox.resize(start_offset);

        if (m_keepTopk > -1) {
            auto k = static_cast<size_t>(m_keepTopk);
            if (num_det > k)
                num_det = k;
        }

        std::vector<int32_t> perm(batchFilteredBox.size());
        std::iota(perm.begin(), perm.end(), 0);

        std::partial_sort(perm.begin(), perm.begin() + num_det, perm.end(), [&batchFilteredBox](int lhs, int rhs) {
            return batchFilteredBox[lhs].score > batchFilteredBox[rhs].score ||
                   (batchFilteredBox[lhs].score == batchFilteredBox[rhs].score && batchFilteredBox[lhs].class_index < batchFilteredBox[rhs].class_index) ||
                   (batchFilteredBox[lhs].score == batchFilteredBox[rhs].score && batchFilteredBox[lhs].class_index == batchFilteredBox[rhs].class_index &&
                    batchFilteredBox[lhs].index < batchFilteredBox[rhs].index);
        });

        auto offset = batch * m_realNumClasses * m_realNumBoxes;
        for (size_t i = 0; i < num_det; i++) {
            m_filteredBoxes[offset + i] = batchFilteredBox[perm[i]];
        }
        m_numPerBatch[batch] = num_det;
    });

    auto start_offset = m_numPerBatch[0];
    for (size_t i = 1; i < m_numPerBatch.size(); i++) {
        auto offset_batch = i * m_realNumClasses * m_realNumBoxes;
        for (size_t j = 0; j < m_numPerBatch[i]; j++) {
            m_filteredBoxes[start_offset + j] = m_filteredBoxes[offset_batch + j];
        }
        start_offset += m_numPerBatch[i];
    }

    m_filteredBoxes.resize(start_offset);
    if (m_sortResultAcrossBatch) { /* sort across batch */
        if (m_sortResultType == MatrixNmsSortResultType::SCORE) {
            parallel_sort(m_filteredBoxes.begin(), m_filteredBoxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (l.score > r.score) || (l.score == r.score && l.batch_index < r.batch_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index && l.index < r.index);
            });
        } else if (m_sortResultType == MatrixNmsSortResultType::CLASSID) {
            parallel_sort(m_filteredBoxes.begin(), m_filteredBoxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (l.class_index < r.class_index) || (l.class_index == r.class_index && l.batch_index < r.batch_index) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score > r.score) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score == r.score && l.index < r.index);
            });
        }
    }

    float* selected_outputs = reinterpret_cast<float*>(getChildEdgesAtPort(NMS_SELECTED_OUTPUTS)[0]->getMemoryPtr()->GetPtr());
    int* selected_indices = reinterpret_cast<int*>(getChildEdgesAtPort(NMS_SELECTED_INDICES)[0]->getMemoryPtr()->GetPtr());
    int* valid_outputs = reinterpret_cast<int*>(getChildEdgesAtPort(NMS_VALID_OUTPUTS)[0]->getMemoryPtr()->GetPtr());
    std::copy(m_numPerBatch.begin(), m_numPerBatch.end(), valid_outputs);

    int64_t output_offset = 0;
    int64_t original_offset = 0;
    for (size_t i = 0; i < m_numBatches; i++) {
        auto real_boxes = m_numPerBatch[i];
        valid_outputs[i] = static_cast<int>(real_boxes);

        for (size_t j = 0; j < real_boxes; j++) {
            auto original_index = original_offset + j;
            selected_indices[j + output_offset] = static_cast<int>(m_filteredBoxes[original_index].index);
            auto selected_base = selected_outputs + (output_offset + j) * 6;
            selected_base[0] = m_filteredBoxes[original_index].class_index;
            selected_base[1] = m_filteredBoxes[original_index].score;
            selected_base[2] = m_filteredBoxes[original_index].box.x1;
            selected_base[3] = m_filteredBoxes[original_index].box.y1;
            selected_base[4] = m_filteredBoxes[original_index].box.x2;
            selected_base[5] = m_filteredBoxes[original_index].box.y2;
        }
        std::fill_n(selected_outputs + (output_offset + real_boxes) * 6, (m_maxBoxesPerBatch - real_boxes) * 6, -1);
        std::fill_n(selected_indices + (output_offset + real_boxes), m_maxBoxesPerBatch - real_boxes, -1);
        output_offset += m_maxBoxesPerBatch;
        original_offset += real_boxes;
    }
}

void MKLDNNMatrixNmsNode::checkPrecision(const Precision prec, const std::vector<Precision> precList, const std::string name, const std::string type) {
    if (std::find(precList.begin(), precList.end(), prec) == precList.end())
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' " << type << " precision: " << prec;
}

void MKLDNNMatrixNmsNode::checkOutput(const SizeVector& dims, const std::vector<Precision> precList, const std::string name, const size_t port) {
    checkPrecision(getOriginalOutputPrecisionAtPort(port), precList, name, outType);
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatrixNmsNode, MatrixNms);
