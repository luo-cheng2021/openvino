// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_center_point_box_format)
{
    std::vector<float> boxes_data = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,   1.0, 1.0,
                                     0.5, 0.4,  1.0, 1.0, 0.5, 10.5,  1.0, 1.0,
                                     0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CENTER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{3, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{3, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_flipped_coordinates)
{
    std::vector<float> boxes_data = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                     1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{3, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{3, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_identical_boxes)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                     0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 10, 4};
    const auto scores_shape = Shape{1, 1, 10};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{1, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{1, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_limit_output_size)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{2, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{2, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {2};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_single_box)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 1, 4};
    const auto scores_shape = Shape{1, 1, 1};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{1, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{1, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_by_IOU)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{3, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{3, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_by_IOU_and_scores)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.4f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{2, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{2, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {2};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_two_batches)
{
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
        0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
        0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{2, 6, 4};
    const auto scores_shape = Shape{2, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{4, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{4, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0};
    std::vector<float> expected_selected_scores = {
        0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 1.0, 0.0, 0.95, 1.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_two_classes)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{4, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{4, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0};
    std::vector<float> expected_selected_scores = {
        0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 1.0, 0.95, 0.0, 1.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_by_IOU_and_scores_without_constants)
{
    std::vector<float> boxes_data = {0.0f, 0.0f,  1.0f, 1.0f,  0.0f, 0.1f,   1.0f, 1.1f,
                                     0.0f, -0.1f, 1.0f, 0.9f,  0.0f, 10.0f,  1.0f, 11.0f,
                                     0.0f, 10.1f, 1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f};

    std::vector<float> scores_data = {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};

    std::vector<int64_t> max_output_boxes_per_class_data = {1};
    std::vector<float> iou_threshold_data = {0.4f};
    std::vector<float> score_threshold_data = {0.2f};
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    const auto max_output_boxes_per_class = make_shared<op::Parameter>(element::i64, Shape{1});
    const auto score_treshold = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto soft_nms_sigma = make_shared<op::Parameter>(element::f32, Shape{1});

    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_treshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms,
                                   ParameterVector{boxes,
                                                   scores,
                                                   max_output_boxes_per_class,
                                                   iou_threshold,
                                                   score_treshold,
                                                   soft_nms_sigma});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{1, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{1, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    auto backend_max_output_boxes_per_class = backend->create_tensor(element::i64, {1});
    auto backend_iou_threshold = backend->create_tensor(element::f32, {1});
    auto backend_score_threshold = backend->create_tensor(element::f32, {1});
    auto backend_soft_nms_sigma = backend->create_tensor(element::f32, {1});
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);
    copy_data(backend_max_output_boxes_per_class, max_output_boxes_per_class_data);
    copy_data(backend_iou_threshold, iou_threshold_data);
    copy_data(backend_score_threshold, score_threshold_data);
    copy_data(backend_soft_nms_sigma, std::vector<float>(0.0));

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes,
                  backend_scores,
                  backend_max_output_boxes_per_class,
                  backend_iou_threshold,
                  backend_score_threshold,
                  backend_soft_nms_sigma});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3};
    std::vector<float> expected_selected_scores = {0.0f, 0.0f, 0.95f};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_reference)
{

    std::vector<float> boxes_data = {
            0.2369726, 0.28203848, 0.51645845, 0.539889,
            0.4985257, 0.49854738, 0.70052814, 0.5731168,
            0.24534181, 0.20871444, 0.6978695, 0.9007864,
            0.3640467, 0.18057947, 0.88013, 0.70254624,
            0.44733843, 0.08509488, 0.9813876, 0.51019615,
            0.0800431, 0.4405484, 0.8949218, 0.7538091,
            0.39023185, 0.2947409, 0.5822403, 0.6811248,
            0.43342504, 0.47589797, 0.8638232, 0.5663469,
            0.09810594, 0.34307447, 0.58732116, 0.63525164,
            0.02561388, 0.3969397, 0.836488, 0.6961541,
            0.38046527, 0.0203846, 0.8423643, 0.70654124,
            0.32604268, 0.48462036, 0.93912554, 0.76133525,
            0.28386533, 0.07562917, 0.72319627, 0.71275604,
            0.33422744, 0.16976027, 0.5971304, 0.897465,
            0.4521741, 0.43520954, 0.8457792, 0.96897304,
            0.41536444, 0.22515537, 0.5683615, 0.5188405,
            0.23329352, 0.3111817, 0.9340813, 0.9539974,
            0.39396682, 0.0682839, 0.9657403, 0.89008236,
            0.39679193, 0.35343063, 0.6144743, 0.7443275,
            0.11305821, 0.47204477, 0.76243985, 0.62834394
    };

    std::vector<float> scores_data = {
            0.59946257, 0.6581636, 0.48800406, 0.44284198, 0.60442245, 0.48058632,
            0.4957834, 0.38717481, 0.48625258, 0.4617983,
            0.40053746, 0.34183636, 0.5119959, 0.55715805, 0.39557755, 0.5194137,
            0.5042166, 0.61282516, 0.5137474, 0.5382017,
            0.66382104, 0.56581205, 0.5002907, 0.5758071, 0.6055713, 0.4621192,
            0.5324077, 0.5875278, 0.48963425, 0.4790761,
            0.33617902, 0.43418795, 0.49970928, 0.42419288, 0.3944287, 0.53788084,
            0.46759236, 0.41247222, 0.5103657, 0.5209239
    };

    std::vector<int64_t> max_output_boxes_per_class_data = {40};
    std::vector<float> iou_threshold_data = {0.4f};
    std::vector<float> score_threshold_data = {0.01f};
    const auto box_encoding = op::v8::MatrixNms::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{2, 10, 4};
    const auto scores_shape = Shape{2, 2, 10};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    const auto max_output_boxes_per_class = make_shared<op::Parameter>(element::i64, Shape{1});
    const auto score_threshold = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto iou_threshold = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto soft_nms_sigma = make_shared<op::Parameter>(element::f32, Shape{1});

    auto nms = make_shared<op::v8::MatrixNms>(boxes,
                                              scores,
                                              max_output_boxes_per_class,
                                              iou_threshold,
                                              score_threshold,
                                              soft_nms_sigma,
                                              box_encoding,
                                              false);
    auto f = make_shared<Function>(nms,
                                   ParameterVector{boxes,
                                                   scores,
                                                   max_output_boxes_per_class,
                                                   iou_threshold,
                                                   score_threshold,
                                                   soft_nms_sigma});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{20, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{20, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    auto backend_max_output_boxes_per_class = backend->create_tensor(element::i64, {1});
    auto backend_iou_threshold = backend->create_tensor(element::f32, {1});
    auto backend_score_threshold = backend->create_tensor(element::f32, {1});
    auto backend_soft_nms_sigma = backend->create_tensor(element::f32, {1});
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);
    copy_data(backend_max_output_boxes_per_class, max_output_boxes_per_class_data);
    copy_data(backend_iou_threshold, iou_threshold_data);
    copy_data(backend_score_threshold, score_threshold_data);
    copy_data(backend_soft_nms_sigma, std::vector<float>(0.0));

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes,
                  backend_scores,
                  backend_max_output_boxes_per_class,
                  backend_iou_threshold,
                  backend_score_threshold,
                  backend_soft_nms_sigma});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    for( auto& i : selected_indeces_value)
        std::cout << i << endl;

}
