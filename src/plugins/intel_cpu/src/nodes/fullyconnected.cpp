// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.h"

#include "eltwise.h"
#include "ie_precision.hpp"
#include "ie_system_conf.h"
#include "input.h"
#include "fake_quantize.h"
#include "input.h"
#include "memory_desc/blocked_memory_desc.h"
#include "reorder.h"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"

#include "onednn/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "common/primitive_hashing_utils.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/reorder_prim.h"
#include "ie_parallel.hpp"
#include "common/dnnl_thread.hpp"

#include <string>
#include <vector>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct FCKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    dnnl::primitive_attr attr;
    impl_desc_type implType;
    bool useConv1x1;

    size_t hash() const;
    bool operator==(const FCKey& rhs) const;
};

size_t FCKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    seed = hash_combine(seed, useConv1x1);
    return seed;
}

bool FCKey::operator==(const FCKey &rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }
    retVal = retVal && *attr.get() == *rhs.attr.get() &&
             implType == rhs.implType && useConv1x1 == rhs.useConv1x1;
    return retVal;
}

class FCShapeInfer : public ShapeInferEmptyPads {
public:
    FCShapeInfer(size_t outPut_rank) : out_rank(outPut_rank) {}
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const VectorDims& activationShape = input_shapes[0].get();
        const VectorDims& weightShape = input_shapes[1].get();
        size_t activationRank = activationShape.size();
        size_t channelRank = weightShape.size() - 1;

        // activation   weight    output_shape
        // NCHW         CoCHW     NCo
        // TNC          CoC       TNCo
        // NC           CoC       NCo
        VectorDims outputShape(out_rank, 1);
        // set Co
        outputShape.back() = weightShape[0];
        // set batch dims
        size_t batchRank = activationRank - channelRank;
        size_t startIdx = out_rank - batchRank - 1;
        for (size_t i = 0; i < batchRank; i++) {
            outputShape[i + startIdx] = activationShape[i];
        }

        return {{std::move(outputShape)}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    size_t out_rank = 0;
};

class FCShapeInferFactory : public ShapeInferFactory {
public:
    FCShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<FCShapeInfer>(m_op->get_output_partial_shape(0).rank().get_length());
    }

private:
    std::shared_ptr<const ngraph::Node> m_op;
};

} // namespace

bool FullyConnected::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto fc = std::dynamic_pointer_cast<const FullyConnectedNode>(op);
        if (!fc) {
            errorMessage = "Only legacy FullyConnected operation is supported";
            return false;
        }
        if (fc->get_input_size() == 3 && std::dynamic_pointer_cast<const ngraph::opset1::Constant>(fc->get_input_node_shared_ptr(BIAS_ID)) == nullptr) {
            errorMessage = "Only Constant operation on 'bias' input is supported";
            return false;
        }
        const auto inRank = fc->get_input_partial_shape(DATA_ID).size();
        const auto weightRank = fc->get_input_partial_shape(WEIGHTS_ID).size();
        if (!one_of(inRank, 2u, 3u, 4u)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank);
            return false;
        }
        if ((one_of(inRank, 2u, 3u) && weightRank != 2) || (inRank == 4 && weightRank != 4)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank) +
                           " and 'weight' input with rank: " + std::to_string(weightRank);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

FullyConnected::FullyConnected(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, FCShapeInferFactory(op)), withBiases(false) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    errorPrefix = "FullyConnected node with name '" + getName() + "'";
    if (context->getConfig().fcSparseWeiDecompressionRate < 1.0f)
        minSparseRate = context->getConfig().fcSparseWeiDecompressionRate;

    expectedBiasDims = {getInputShapeAtPort(WEIGHTS_ID).getStaticDims()[0]};

#ifdef OV_CPU_WITH_LLMDNN
    const std::string& pkey = ov::PrimitivesPriority::get_type_info_static();
    const auto& info = op->get_rt_info();
    if (info.count(pkey)) {
        auto primitives_priority_attr = info.at(pkey).as<ov::PrimitivesPriority>().value;
        if (primitives_priority_attr.find("gemm_llmdnn") == std::string::npos)
            stateLLMFc = State_NotUse;
    }
#endif
}

std::vector<memory::format_tag> FullyConnected::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return {memory::format_tag::tnc};
    else if (dims.getRank() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.getRank() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

VectorDims FullyConnected::makeDummyInputDims() const {
    const auto& inShape = getInputShapeAtPort(DATA_ID);
    const auto& weightDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();

    auto inMinDims = inShape.getMinDims();
    auto inMaxDims = inShape.getMaxDims();

    if (inMinDims.size() == 3) {
        inMinDims.back() = weightDims.back();
        inMaxDims.back() = weightDims.back();
    } else {
        for (size_t i = 1; i < inMinDims.size(); i++) {
            inMinDims[i] = weightDims[i];
            inMaxDims[i] = weightDims[i];
        }
    }
    return MemoryDescUtils::makeDummyShape(Shape(inMinDims, inMaxDims)).getStaticDims();
}

VectorDims FullyConnected::makeDummyOutputDims(const VectorDims& inDims) const {
    std::vector<Shape> inShapes = {Shape(inDims), getInputShapeAtPort(WEIGHTS_ID)};
    if (inputShapes.size() > 2) {
        inShapes.emplace_back(getInputShapeAtPort(BIAS_ID));
    }
    return shapeInferGeneric(inShapes).front();
}

bool FullyConnected::canBeExecutedInInt8() const {
    auto firstInputPrecision = getOriginalInputPrecisionAtPort(0);
    auto secondInputPrecision = getOriginalInputPrecisionAtPort(1);

    return one_of(firstInputPrecision, Precision::U8, Precision::I8) && secondInputPrecision == Precision::I8;
}

void FullyConnected::getSupportedDescriptors() {
    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW()<< errorPrefix << " has incorrect number of output edges";

    withBiases = getOriginalInputsNumber() == 3;

    useSparseWeights = useSparseWeightsDecompression();

    inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(DATA_ID));
    outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(DATA_ID));

    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }
    auto weightsDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(WEIGHTS_ID));
    // revert back outputDataType on special cases
    if (inputDataType == memory::data_type::f32) {
        // oneDNN only support f32 output when input is f32, even if FQ is fused
        outputDataType = memory::data_type::f32;
    } else if (inputDataType == memory::data_type::bf16) {
        // bf16 input only supports bf16/f32 output, even if FQ is fused as post-ops
        if (one_of(outputDataType , memory::data_type::u8, memory::data_type::s8)) {
            outputDataType = memory::data_type::bf16;
        }
    } else if (inputDataType == memory::data_type::f16) {
        // f16 input only supports f16/f32 output, even if FQ is fused as post-ops
        if (!one_of(outputDataType , memory::data_type::f32, memory::data_type::f16)) {
            outputDataType = memory::data_type::f16;
        }
    } else if (one_of(inputDataType, memory::data_type::u8, memory::data_type::s8)) {
        if (weightsDataType != memory::data_type::s8) {
            // weight has to be s8 for INT8 mode, otherwise fallback to
            // f32 mode
            inputDataType = outputDataType = memory::data_type::f32;
        } else if (one_of(outputDataType, memory::data_type::f16)) {
            // INT8 inner-product only supports u8/s8/s32/f32/bf16,
            // other precision needs fallback to f32
            outputDataType = memory::data_type::f32;
        }
    } else {
        // s32/u32/... unsupported input data types, fallback to f32
        inputDataType = outputDataType = memory::data_type::f32;
    }

    inDims = isDynamicNode() ? makeDummyInputDims() : getInputShapeAtPort(DATA_ID).getStaticDims();
    outDims = isDynamicNode() ? makeDummyOutputDims(inDims) : getOutputShapeAtPort(0).getStaticDims();

    for (auto format : getAvailableFormatsForDims(getInputShapeAtPort(0))) {
        auto in_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(inDims), inputDataType, format);
        auto out_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(outDims), outputDataType, dnnl::memory::format_tag::any);

        createDescriptorInternal(in_candidate, out_candidate);
    }
}

void FullyConnected::createPrimitive() {
#ifdef OV_CPU_WITH_LLMDNN
    if (stateLLMFc == State_Use)
        return;
#endif
    setPostOps(attr, outDims);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    Node::createPrimitive();
    appendPostOpArgs(attr, primArgs, postOpsArgs);
}

void FullyConnected::prepareParams() {
#ifdef OV_CPU_WITH_LLMDNN
    if (tryUseLLMFc())
        return;
#endif

    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory hasn't been allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory hasn't been allocated.";
    MemoryPtr biasMemPtr = nullptr;
    if (withBiases) {
        biasMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
        if (!biasMemPtr || !biasMemPtr->isAllocated())
            IE_THROW() << "Input memory hasn't been allocated.";
    }

    NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

    DnnlMemoryDescPtr weightDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightDescIP);
    DnnlMemoryDescCPtr biasDesc = nullptr;
    if (biasMemPtr) {
        biasDesc = biasMemPtr->GetDescWithType<DnnlMemoryDesc>();
    }

    DnnlMemoryDescCPtr inDesc = srcMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr outDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();

    useConv1x1 = canBeExecutedInConv1x1();
    FCKey key = {inDesc,
                 weightDesc,
                 biasDesc,
                 outDesc,
                 attr,
                 implementationTypeIP,
                 useConv1x1};

    auto& engine = getEngine();

    auto builder = [&engine](const FCKey& key) -> executorPtr {
        // use conv1x1 primitive for computation
        if (key.useConv1x1) {
            auto prim_desc = createDescriptorInternalForConv(key.inp0, key.inp1, key.bias, key.out, key.attr, engine);
            const bool found = DnnlExtensionUtils::find_implementation(prim_desc, brgconv_avx512_1x1);

            if (found)
                return std::make_shared<DnnlExecutor>(prim_desc);
        }

        // fallback to normal convolution primitive
        auto inDesc = key.inp0->getDnnlDesc();
        const auto& inDims = inDesc.get_dims(); // @TODO query + copy might be slow
        if (inDims.size() == 3) {
            auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};
            inDesc = inDesc.reshape(normalizedInDims);
        }
        auto outDesc = key.out->getDnnlDesc();
        const auto& outDims = outDesc.get_dims(); // @TODO query + copy might be slow

        if (outDims.size() == 3) {
            auto normalizedOutDims = { outDims[0] * outDims[1], outDims[2] };
            outDesc = outDesc.reshape(normalizedOutDims);
        }

        dnnl::inner_product_forward::primitive_desc prim_desc;
        if (key.bias) {
            prim_desc = dnnl::inner_product_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                inDesc,
                key.inp1->getDnnlDesc(),
                key.bias->getDnnlDesc(),
                outDesc,
                key.attr);
        } else {
            prim_desc = dnnl::inner_product_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                inDesc,
                key.inp1->getDnnlDesc(),
                outDesc,
                key.attr);
        }

        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, key.implType);

        if (!found)
            return nullptr;

        return std::make_shared<DnnlExecutor>(prim_desc);;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    auto prevExecPtr = execPtr;
    execPtr = result.first;

    if (execPtr) {
        if (execPtr->getSrcDesc()->isCompatible(*inDesc)) {
            primArgs[DNNL_ARG_SRC] = srcMemPtr->GetPrimitive();
        } else {
            primArgs[DNNL_ARG_SRC] = dnnl::memory(execPtr->getDnnlSrcDesc(), engine, srcMemPtr->GetData());
        }

        if (execPtr->getDstDesc()->isCompatible(*outDesc)) {
            primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();
        } else {
            primArgs[DNNL_ARG_DST] = dnnl::memory(execPtr->getDnnlDstDesc(), engine, dstMemPtr->GetData());
        }

        if (!prevExecPtr || !execPtr->getWeightDesc()->isCompatible(*(prevExecPtr->getWeightDesc()))) {
            primArgs[DNNL_ARG_WEIGHTS] = prepareWeightMemory(execPtr->getWeightDesc())->GetPrimitive();
        }
        // changed shapes may also cause the kernel type changed
        selected_pd->setImplementationType(execPtr->getImplementationType());
        // WA: We update implType to know whether weights decompression was used inside the kernel
        if (selected_pd->getImplementationType() == ov::intel_cpu::brgemm_avx512_amx && useSparseWeights) {
            selected_pd->setImplementationType(ov::intel_cpu::brgemm_sparse_avx512_amx);
        }
        // maybe expected 1x1 conv is not created, update the flag depends on the real type
        useConv1x1 = execPtr->getImplementationType() == brgconv_avx512_1x1;

        if (withBiases) {
            primArgs[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
        }

        auto schratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());
        primArgs[DNNL_ARG_SCRATCHPAD] = schratchpadMem->GetPrimitive();
#ifdef CPU_DEBUG_CAPS
        if (result.second == CacheEntryBase::LookUpStatus::Miss) {
            auto pd = execPtr->getPrimitiveDesc();
            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
        }
#endif
    } else {
        IE_THROW() << "Executor is not created for node " << getName() << ".";
    }
}

void FullyConnected::execute(dnnl::stream strm) {
#ifdef OV_CPU_WITH_LLMDNN
    if (tryExecLLMFc())
        return;
#endif

    if (!execPtr) {
        IE_THROW() << "Can't execute FullyConnected node with name: " << getName() << ", because executor is not compiled";
    }

    // in cases parameter -> FullyConnected or dynamic shapes
    // we keep old pointer to data in primArgs on second iteration with same input shapes
    auto updateMemoryPtr = [this](int argType) {
        auto param = primArgs.find(argType);
        if (param != primArgs.end()) {
            if (argType == DNNL_ARG_SRC && (getInputShapeAtPort(DATA_ID).getRank() == 3 || useConv1x1)) {
                primArgs.at(argType).set_data_handle(getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetData());
            }
            if (argType == DNNL_ARG_DST && (getOutputShapeAtPort(0).getRank() == 3 || useConv1x1)) {
                primArgs.at(argType).set_data_handle(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetData());
            }
        }
    };

    updateMemoryPtr(DNNL_ARG_SRC);
    updateMemoryPtr(DNNL_ARG_DST);

    execPtr->exec(primArgs, strm);
}

void FullyConnected::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool FullyConnected::canFuse(const NodePtr& node) const {
    return canFuseSimpleOperation(node);
}

void FullyConnected::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims_ext) {
    dnnl::post_ops ops;

    // accoridng to https://oneapi-src.github.io/oneDNN/dev_guide_inner_product.html
    // oneDNN inner product primitive's input & output tensors are always 2D:
    //   input: [N, IC]  weight: [OC, IC]   bias: [OC]   output:[N,OC]
    //
    // when input output tensors have spatial dimensions, they are flattened to 2D.
    // and following type of MatMul will be converted into FullyConnected inside CPU plugin:
    //    2D:   [X,Y] [Y,Z] =>   [X,Z]   with    N=X,IC=Y,OC=Z
    //    3D: [B,X,Y] [Y,Z] => [B,X,Z]   with  N=B*X,IC=Y,OC=Z

    VectorDims dims;
    if (dims_ext.size() == 2) {
        // 2D
        dims = dims_ext;
    } else if (dims_ext.size() == 3) {
        // 3D
        dims.push_back(dims_ext[0] * dims_ext[1]);
        dims.push_back(dims_ext[2]);
    } else {
        IE_THROW() << "Unexpected rank(" << dims_ext.size() << ") for output tensor of node: " << getName();
    }

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, canBeExecutedInInt8(),
                                    1 << 0,  getDQScales(), withBiases);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

bool FullyConnected::created() const {
    return getType() == Type::FullyConnected;
}

const std::vector<impl_desc_type>& FullyConnected::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::acl,
        impl_desc_type::brgemm_sparse_avx512_amx,
        impl_desc_type::brgemm_avx512_amx,
        impl_desc_type::brgemm_avx512,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,
        impl_desc_type::gemm_sse42,
        impl_desc_type::gemm_any,
        impl_desc_type::gemm,
        impl_desc_type::jit_gemm,
        impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,
        impl_desc_type::jit_uni,
        impl_desc_type::jit_avx512_dw,
        impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,
        impl_desc_type::jit_avx2_dw,
        impl_desc_type::jit_avx2_1x1,
        impl_desc_type::jit_avx2,
        impl_desc_type::jit_avx_dw,
        impl_desc_type::jit_avx_1x1,
        impl_desc_type::jit_avx,
        impl_desc_type::jit_sse42_dw,
        impl_desc_type::jit_sse42_1x1,
        impl_desc_type::jit_sse42,
        impl_desc_type::ref,
    };

    return priorities;
}

// WA: creation DnnlMemoryDesc with format == any is prohibited
// so we create dnnl::memory::desc directly
// we need specific method and can't remove createDescriptor from base class because its used into initDescriptor
void FullyConnected::createDescriptorInternal(const dnnl::memory::desc &inputDesc,
                                              const dnnl::memory::desc &outputDesc) {
    auto create2Dcandidate = [](const dnnl::memory::desc &desc) {
        if (desc.get_dims().size() != 3) // already 2D
            return desc;

        auto inDims = desc.get_dims();
        auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};

        return dnnl::memory::desc(normalizedInDims, desc.get_data_type(),
                                  DnnlExtensionUtils::GetPlainFormatByRank(normalizedInDims.size()));
    };

    const auto in_candidate  = create2Dcandidate(inputDesc);
    const auto out_candidate = create2Dcandidate(outputDesc);

    const dnnl::memory::data_type indt = inputDesc.get_data_type();
    const dnnl::memory::data_type outdt = outputDesc.get_data_type();
    dnnl::memory::data_type wdt = indt;
    dnnl::memory::data_type bdt = outdt;

    if (one_of(indt, dnnl::memory::data_type::bf16, dnnl::memory::data_type::f16)) {
        bdt = dnnl::memory::data_type::f32;
    } else if (indt == dnnl::memory::data_type::u8 || indt == dnnl::memory::data_type::s8) {
        wdt = memory::data_type::s8;
        if (withBiases)
            bdt = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(BIAS_ID));
    }
    // We need to explicitly specify the memory descriptor to use sparse weights decompression
    dnnl::memory::desc wgh_candidate;
    if (useSparseWeights) {
        wgh_candidate = wgh_candidate.sparse_desc(DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                                                  wdt);
    } else {
        wgh_candidate = { DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                                        wdt, dnnl::memory::format_tag::any };
    }

    const dnnl::primitive_attr attr;

    if (withBiases) {
        dnnl::memory::desc bias_candidate(DnnlExtensionUtils::convertToDnnlDims(expectedBiasDims), bdt,
                                            dnnl::memory::format_tag::any);
        auto desc = inner_product_forward::primitive_desc(
            getEngine(),
            prop_kind::forward_inference,
            in_candidate,
            wgh_candidate,
            bias_candidate,
            out_candidate,
            attr,
            true);

        descs.push_back(desc);
    } else {
        auto desc = inner_product_forward::primitive_desc(
            getEngine(),
            prop_kind::forward_inference,
            in_candidate,
            wgh_candidate,
            out_candidate,
            attr,
            true);

        descs.push_back(desc);
    }
}

void FullyConnected::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                      const std::vector<MemoryDescPtr> &outputDesc) {
    MemoryDescPtr inpDesc;
    if (inputDesc[0]->isDefined()) {
        inpDesc = inputDesc[0];
    } else {
        inpDesc = inputDesc[0]->cloneWithNewDims(inDims);
    }

    MemoryDescPtr outDesc;
    if (outputDesc[0]->isDefined()) {
        outDesc = outputDesc[0];
    } else {
        outDesc = outputDesc[0]->cloneWithNewDims(outDims);
    }
    createDescriptorInternal(MemoryDescUtils::convertToDnnlMemoryDesc(inpDesc)->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(outDesc)->getDnnlDesc());
}

void FullyConnected::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // 3D FC requires implicit reshape so strides should be defined
    auto supportsUndefStridesAndOffset = [&]() {
        return getOutputShapeAtPort(0).getRank() == 2;
    };

    auto addSupportedPrimitiveDescriptor = [&](const dnnl::primitive_desc& prim_desc) {
        std::vector<PortConfig> inConfs, outConfs;
        const int inPlaceOutPort = canBeInPlace() ? 0 : -1;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            auto desc = getSrcMemDesc(prim_desc, i);
            const auto inputBlockedMask = (supportsUndefStridesAndOffset() && !(i == WEIGHTS_ID && useSparseWeights)) ?
                BlockedMemoryDesc::EMPTY_MASK :
                BlockedMemoryDesc::FULL_MASK;

            inConfs.emplace_back(desc, inputBlockedMask);
        }

        const auto outputBlockedMask = supportsUndefStridesAndOffset() ? BlockedMemoryDesc::EMPTY_MASK : BlockedMemoryDesc::FULL_MASK;

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            auto desc = getDstMemDesc(prim_desc, i);

            outConfs.emplace_back(desc, outputBlockedMask, inPlaceOutPort);
        }

        const NodeConfig config(inConfs, outConfs);
        const impl_desc_type impl_type = parse_impl_name(prim_desc.impl_info_str());

        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    };

    for (auto& desc : descs) {
        auto first_desc = dnnl::primitive_desc(DnnlExtensionUtils::clone_primitive_desc(desc.get()));

        const bool first_match = customImplPriorities.empty();
        DnnlExtensionUtils::for_each_implementation(desc,
                                                    first_match,
                                                    [&](impl_desc_type implType) {
                                                        return contains(getImplPriority(), implType);
                                                    },
                                                    [&](dnnl::primitive_desc& desc) {
                                                        addSupportedPrimitiveDescriptor(desc);
                                                    });

        // fallback. if none of the primitive types is present in the priority list just add first implementation
        // @todo this fallback is not necessary if primitive priority list is filled correctly
        if (supportedPrimitiveDescriptors.empty())
            addSupportedPrimitiveDescriptor(first_desc);
    }
}

std::shared_ptr<MemoryDesc> FullyConnected::getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc = idx > 0 ? prim_desc.weights_desc(idx - 1) : prim_desc.src_desc(idx);

    if (getInputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(desc.get_data_type()), getInputShapeAtPort(idx));
    }

    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }

    return DnnlExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> FullyConnected::getDstMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc = prim_desc.dst_desc(idx);

    if (getOutputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(desc.get_data_type()), getOutputShapeAtPort(idx));
    }

    if (getOutputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getOutputShapeAtPort(idx));
    }

    return DnnlExtensionUtils::makeDescriptor(desc);
}

InferenceEngine::Precision FullyConnected::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

void FullyConnected::initOptimalPrimitiveDescriptor() {
    Node::initOptimalPrimitiveDescriptor();
    auto selectedPD = getSelectedPrimitiveDescriptor();
    implementationTypeIP = selectedPD->getImplementationType();
    // if convolution selected the reorder for ip is useless. Will do the reoder for ip in prepareParams
    auto constParent = getParentEdgeAt(1)->getParent();
    auto selectedParentPD = constParent->getSelectedPrimitiveDescriptor();
    auto config = selectedPD->getConfig();
    weightDescIP = config.inConfs[1].getMemDesc();
    config.inConfs[1].setMemDesc(selectedParentPD->getConfig().outConfs[0].getMemDesc());
    selectedPD->setConfig(config);
}

dnnl::convolution_forward::primitive_desc
FullyConnected::createDescriptorInternalForConv(DnnlMemoryDescCPtr inputDescPtr,
                                                DnnlMemoryDescCPtr weightDescPtr,
                                                DnnlMemoryDescCPtr biasDescPtr,
                                                DnnlMemoryDescCPtr outputDescPtr,
                                                const dnnl::primitive_attr& attr,
                                                const dnnl::engine& engine) {
    const dnnl::memory::desc &inputDesc  = inputDescPtr->getDnnlDesc();
    const dnnl::memory::desc &outputDesc = outputDescPtr->getDnnlDesc();
    const dnnl::memory::desc &weightDesc = weightDescPtr->getDnnlDesc();
    // make a fake shape: N, IC, W
    auto inDims = inputDesc.get_dims();
    dnnl::memory::dims normalizedInDims;
    if (inDims.size() == 3) {
        normalizedInDims = {inDims[0], inDims[2], inDims[1]};
    } else if (inDims.size() == 2) {
        normalizedInDims = {dnnl::memory::dim{1}, inDims[1], inDims[0]};
    }
    auto convInDesc = dnnl::memory::desc(normalizedInDims, inputDesc.get_data_type(), memory::format_tag::nwc);

    // make a fake shape: N, OC, W
    const auto& outDims = outputDesc.get_dims();
    dnnl::memory::dims normalizedOutDims;
    if (outDims.size() == 3) {
        normalizedOutDims = { outDims[0], outDims[2], outDims[1]};
    } else if (outDims.size() == 2) {
        normalizedOutDims = { dnnl::memory::dim{1}, outDims[1], outDims[0]};
    }
    auto convOutDesc = dnnl::memory::desc(normalizedOutDims, outputDesc.get_data_type(), memory::format_tag::nwc);

    // make a fake shape: OC, IC, 1
    auto weightDims = weightDesc.get_dims();
    dnnl::memory::dims normalizedWeightDims;
    normalizedWeightDims = {static_cast<dnnl::memory::dim>(weightDims[0]),
                            static_cast<dnnl::memory::dim>(weightDims[1]),
                            dnnl::memory::dim{1}};
    auto convWeightDescAny = dnnl::memory::desc(normalizedWeightDims, weightDesc.get_data_type(), dnnl::memory::format_tag::any);

    if (biasDescPtr) {
        return dnnl::convolution_forward::primitive_desc(
            engine,
            prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            convInDesc, convWeightDescAny, biasDescPtr->getDnnlDesc(), convOutDesc,
            dnnl::memory::dims{1},   // stride
            dnnl::memory::dims{0},   // dilation
            dnnl::memory::dims{0},   // paddingL
            dnnl::memory::dims{0},   // paddingR
            attr);
    } else {
        return dnnl::convolution_forward::primitive_desc(
            engine,
            prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
            convInDesc, convWeightDescAny, convOutDesc,
            dnnl::memory::dims{1},   // stride
            dnnl::memory::dims{0},   // dilation
            dnnl::memory::dims{0},   // paddingL
            dnnl::memory::dims{0},   // paddingR
            attr);
    }
}

bool FullyConnected::canBeExecutedInConv1x1() const {
    bool retVal = false;
    const auto inRank = getInputShapeAtPort(DATA_ID).getRank();
    const auto weightRank = getInputShapeAtPort(WEIGHTS_ID).getRank();
    // disable rank=4:
    // if layout is nhwc:
    //   A matrix: N * IC * H * W --> N * (IC*H*W), the M, N', K of matrix multiply will be:
    //   M = 1, K = (IC*H*W), when M = 1 it should not be efficient since acts as a vector multiply
    // if layout is nchw/nChw16c: brg1x1 not support. Although jit supports, it should have similar
    //   problems with the above.
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
        getOriginalInputPrecisionAtPort(DATA_ID) == InferenceEngine::Precision::FP32 &&
        one_of(inRank, 2u, 3u) && weightRank == 2) {
        auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
        DnnlMemoryDescCPtr outDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();
        // brg convolution does not support stride
        dnnl::impl::memory_desc_wrapper wrapped(outDesc->getDnnlDesc().get());
        if (wrapped.offset0() == 0)
            retVal = true;
    }

    if (retVal) {
        auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
        const auto& srcDims = srcMemPtr->getStaticDims();
        auto weightMemPtr = getParentEdgesAtPort(1)[0]->getMemoryPtr();
        const auto& weightDims = weightMemPtr->getStaticDims();
        // for original inner product semantics:
        //  when input is 2D tensor
        //    M in oneDNN will map to widthInConv
        //  when input is 3D tensor
        //    M in oneDNN will map to widthInConv*minibatch
        // currently nwc mapping in brg:
        //  when input is 2D tensor
        //    widthInConv will map to 'w', 'n' will be 1
        //  when input is 3D tensor
        //    widthInConv will map to 'w', 'n' will be minibatch
        Dim widthInConv, N, K;
        widthInConv = srcDims[inRank - 2];
        K = srcDims[inRank - 1];
        N = weightDims[0];

        if (!(widthInConv >= 2 && widthInConv <= 3136 &&
              K >= 96 && K <= 4096 &&
              N >= 96 && N <= K * 4))
            retVal = false;
    }

    return retVal;
}

bool FullyConnected::useSparseWeightsDecompression() {
    // minSparseRate == 1 means that sparse feature is switched off
    if (minSparseRate == 1.f) {
        return false;
    }

    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx))
        return false;

    auto weiDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();
    if (weiDims.size() != 2 || weiDims[0] % 64 != 0 || weiDims[1] % 64 != 0) {
        return false;
    }

    auto inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    auto weightsPrecision = getOriginalInputPrecisionAtPort(WEIGHTS_ID);
    if (!one_of(inputPrecision , Precision::U8, Precision::I8) || weightsPrecision != Precision::I8) {
        return false;
    }

    // calculate sparse rate
    const auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgeAt(WEIGHTS_ID)->getParent());
    if (!constNode) {
        return false;
    }
    auto blb = constNode->getMemoryPtr();
    if (blb == nullptr)
        IE_THROW() << "Cannot get const blob for node " << getName() << ".";

    auto weightsData = reinterpret_cast<const int8_t*>(blb->GetPtr());
    auto elementsCount = blb->GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    size_t zerosCounts = 0;
    for (size_t i = 0; i < elementsCount; i++) {
        if (weightsData[i] == 0) {
            zerosCounts++;
        }
    }

    DEBUG_LOG(getName(), ", elementsCount = ", elementsCount, ", zerosCounts = ",
        zerosCounts, ", nnzCount = ", elementsCount - zerosCounts);

    weiSparseRate = static_cast<float>(zerosCounts) / static_cast<float>(elementsCount);

    DEBUG_LOG(getName(), " | sparse rate = ", weiSparseRate * 100, "%, min sparse rate = ",
        minSparseRate * 100, "%, use sparse weights = ", weiSparseRate >= minSparseRate);

    if (weiSparseRate < minSparseRate) {
        return false;
    }

    return true;
}

#ifdef OV_CPU_WITH_LLMDNN
bool FullyConnected::tryExtractParamForLLMFc(llmdnn::fc_create_param& param) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx))
        return false;

    if (useSparseWeights)
        return false;

    const auto inRank = getInputShapeAtPort(DATA_ID).getRank();
    if (!one_of(inRank, 2u, 3u))
        return false;

    const auto& weight_dims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();
    const auto N = weight_dims[0];
    const auto K = weight_dims[1];
    const auto data_type = getOriginalInputPrecisionAtPort(DATA_ID);
    // heuristics
    if ((data_type == Precision::BF16 && K < 32) ||
        (data_type == Precision::I8 && K < 64) ||
        // TODO: add int8 support
        (data_type != Precision::BF16) ||
        (!isDynamicNode()) ||
        // 1 stream on 1+ numa node. Limitation: weights do not share among with multiple
        //   streams inside a numa because LLM will run with few streams.
        (context->getConfig().streamExecutorConfig._streams > get_num_numa_nodes()))
        return false;

    auto tryExtractBias = [&] () {
        auto* bias = reinterpret_cast<float*>(getParentEdgeAt(BIAS_ID)->getMemoryPtr()->GetPtr());
        auto bias_count = getParentEdgeAt(BIAS_ID)->getMemoryPtr()->GetShape().getElementsCount();
        auto capacity = rnd_up(N * sizeof(float), 64);
        biasRnd = std::shared_ptr<float>(
                            reinterpret_cast<float*>(aligned_alloc(64, capacity)),
                            [](void * p) { ::free(p); });
        memset(biasRnd.get(), 0, capacity);

        if (bias_count == 1) {
            std::fill(biasRnd.get(), biasRnd.get() + N, bias[0]);
        } else {
            memcpy(biasRnd.get(), bias, N * sizeof(float));
        }
    };
    if (data_type == Precision::BF16) {
        if (one_of(outputDataType, memory::data_type::f32, memory::data_type::bf16) &&
            (fusedWith.empty() ||
            (fusedWith.size() == 1 && (fusedWith[0]->getAlgorithm() == Algorithm::EltwiseGeluErf ||
                                       fusedWith[0]->getAlgorithm() == Algorithm::EltwiseGeluTanh)))) {
            param.dt_a = llmdnn::dnnl_bf16;
            param.dt_b = llmdnn::dnnl_bf16;
            param.dt_c = static_cast<llmdnn::data_type_t>(outputDataType);
            param.b_is_trans = true;
            param.postops_type = llmdnn::NONE;
            if (withBiases) {
                param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::BIAS);
                tryExtractBias();
            }
            if (fusedWith.size() == 1) {
                if (fusedWith[0]->getAlgorithm() == Algorithm::EltwiseGeluErf)
                    param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::GELU);
                else
                    param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::GELU_TANH);
            }

            // compute q/dq
            auto p = getenv("USE_INT8_WEIGHT");
            if (p && p[0] == '1') {
                auto weightPtr = castMemoryPtr(getParentEdgeAt(WEIGHTS_ID)->getMemoryPtr(), data_type);
                auto* weight = reinterpret_cast<bfloat16*>(weightPtr->GetPtr());
                auto& weight_dims = weightPtr->getStaticDims();
                llmdnn::fc_kernel_bf16w8_get_q_dq(weight_dims[0], weight_dims[1], weight_dims[1] * sizeof(bfloat16), weight, &param.q, &param.dq);
                param.dt_b = llmdnn::dnnl_s8;
                param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::DEQUANT);
            }

            return true;
        }
    } else if (data_type == Precision::I8) {
        if (one_of(outputDataType, memory::data_type::f32, memory::data_type::bf16, memory::data_type::s8)) {
            param.dt_a = llmdnn::dnnl_s8;
            param.dt_b = llmdnn::dnnl_s8;
            param.dt_c = static_cast<llmdnn::data_type_t>(outputDataType);
            param.b_is_trans = true;
            param.postops_type = llmdnn::NONE;

            if (withBiases) {
                param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::BIAS);
                tryExtractBias();
            }
            bool firstGelu = true;
            bool firstDQ = true;
            bool firstQ = true;
            bool valid = true;
            for (size_t i = 0; i < fusedWith.size(); ++i) {
                auto& node = fusedWith[i];
                bool isLastPostOp = (i == (fusedWith.size() - 1));

                if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
                    if (firstGelu && (eltwiseNode->getAlgorithm() == Algorithm::EltwiseGeluErf ||
                                      eltwiseNode->getAlgorithm() == Algorithm::EltwiseGeluTanh)) {
                        firstGelu = false;
                        if (eltwiseNode->getAlgorithm() == Algorithm::EltwiseGeluErf)
                            param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::GELU);
                        else
                            param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::GELU_TANH);
                    } else if (eltwiseNode->getAlgorithm() == Algorithm::EltwiseMultiply && firstDQ) {
                        firstDQ = false;
                        const auto& quant = eltwiseNode->getScales();
                        auto capacity = rnd_up(N * sizeof(float), 64);
                        dequant = std::shared_ptr<float>(
                                            reinterpret_cast<float*>(aligned_alloc(64, capacity)),
                                            [](void * p) { ::free(p); });
                        memset(dequant.get(), 0, capacity);

                        if (quant.size() == 1) {
                            std::fill(dequant.get(), dequant.get() + N, quant[0]);
                        } else {
                            memcpy(dequant.get(), quant.data(), quant.size() * sizeof(float));
                        }
                        param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::DEQUANT);
                    } else {
                        valid = false;
                        break;
                    }
                } else if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
                    if (isLastPostOp && firstQ) {
                        firstQ = false;
                        auto capacity = rnd_up(N * sizeof(float), 64);
                        requant = std::shared_ptr<float>(
                                            reinterpret_cast<float*>(aligned_alloc(64, capacity)),
                                            [](void * p) { ::free(p); });
                        memset(requant.get(), 0, capacity);

                        auto& quant = fakeQuantizeNode->getInputScale();

                        if (quant.size() == 1) {
                            std::fill(requant.get(), requant.get() + N, quant[0]);
                        } else {
                            memcpy(requant.get(), quant.data(), quant.size() * sizeof(float));
                        }
                        param.postops_type = static_cast<llmdnn::postops_types>(param.postops_type | llmdnn::QUANT);
                    } else {
                        valid = false;
                        break;
                    }
                } else {
                    valid = false;
                    break;
                }
            }
            return valid;
        }
    }

    return false;
}

MemoryPtr FullyConnected::castMemoryPtr(MemoryPtr memPtr, const InferenceEngine::Precision prec) {
    auto engine = memPtr->getEngine();
    dnnl::stream engine_stream(engine);

    // get weightPtr and dnnlweightDesc
    auto dnnlMemDesc = memPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
    // desc_ptr
    auto newMemDescPtr = memPtr->getDescPtr()->cloneWithNewPrecision(prec);
    // dnnl_desc_ptr
    DnnlMemoryDescPtr dnnlNewMemPtr = MemoryDescUtils::convertToDnnlMemoryDesc(newMemDescPtr);
    // dnnl_desc
    auto dnnlNewMemDesc = dnnlNewMemPtr->getDnnlDesc();
    MemoryPtr newMemPtr = MemoryPtr(new Memory(engine));
    newMemPtr->Create(dnnlNewMemPtr);

    auto reorder = getReorderPrim(context->getParamsCache(), engine, dnnlMemDesc, dnnlNewMemDesc);
    primArgs = {{DNNL_ARG_SRC, memPtr->GetPrimitive()}, {DNNL_ARG_DST, newMemPtr->GetPrimitive()}};
    reorder.execute(engine_stream, primArgs);
    engine_stream.wait();
    return newMemPtr;
}

void FullyConnected::tryCompressWeightForLLMFc() {
    const auto inPrec = getOriginalInputPrecisionAtPort(DATA_ID);
    bool needCompress = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
        one_of(inPrec, InferenceEngine::Precision::BF16);
    // weight
    // If the input precision is fp32 on SPR, the weight will be cast to bf16 for acceleration. 
    auto weightPtr = needCompress ?
        castMemoryPtr(getParentEdgeAt(WEIGHTS_ID)->getMemoryPtr(), inPrec) :
        getParentEdgeAt(WEIGHTS_ID)->getMemoryPtr();
    weightDims = weightPtr->getStaticDims();
    void* weight = weightPtr->GetPtr();

    // src
    auto& srcPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();
    auto* src = srcPtr->GetPtr();
    auto& data_dims = srcPtr->getStaticDims();

    // dst
    auto* dst = getChildEdgeAt(0)->getMemoryPtr()->GetPtr();

    // M, N, K
    auto M = data_dims[0];
    auto N = weightDims[0];
    auto K = weightDims[1];
    if (data_dims.size() == 3) {
        M *= data_dims[1];
        assert(K == data_dims[2]);
    } else {
        assert(K == data_dims[1]);
    }
    auto work_amount = rnd_up(N, 32) / 32;
    float* bias = nullptr;
    if (withBiases) {
        bias = biasRnd.get();
    }
    auto thread_num_fcllm = fcLLMs.size();
    size_t lda, ldb, ldc;
    lda = K * dnnl::memory::data_type_size(inputDataType);
    ldb = K * dnnl::memory::data_type_size(inputDataType);
    ldc = N * dnnl::memory::data_type_size(outputDataType);
    parallel_for(thread_num_fcllm, [&](size_t tid) {
        size_t start {0}, end {0};
        dnnl::impl::balance211(work_amount, thread_num_fcllm, tid, start, end);
        size_t n0 = start * 32;
        size_t n1 = std::min(end * 32, N);
        if (n0 >= N) return;

        llmdnn::fc_kernel_execute(fcLLMs[tid].get(), src, weight, dst, lda, ldb, ldc, M, N, K, n0, n1,
            dequant.get(), requant.get(), bias);
    });
}

bool FullyConnected::tryUseLLMFc() {
    if (stateLLMFc != Not_Init)
        return stateLLMFc == State_Use;

    llmdnn::fc_create_param param;
    if (!tryExtractParamForLLMFc(param)) {
        stateLLMFc = State_NotUse;
        return false;
    }

    auto thread_num = parallel_get_max_threads();
    fcLLMs.resize(thread_num);
    bool ret = true;
    for (int i = 0; i < thread_num; i++) {
        llmdnn::fc_kernel* fc;
        if (!fc_kernel_create(&fc, &param)) {
            ret = false;
            break;
        }
        fcLLMs[i] = std::shared_ptr<llmdnn::fc_kernel>(fc, [](llmdnn::fc_kernel* p) { fc_kernel_destroy(p); });
    }
    if (ret) {
        stateLLMFc = State_Use;
        NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
        selected_pd->setImplementationType(gemm_llmdnn);
        tryCompressWeightForLLMFc();
    } else {
        // fallback
        stateLLMFc = State_NotUse;
        fcLLMs.clear();
    }

    return ret;
}

bool FullyConnected::tryExecLLMFc() {
    if (stateLLMFc != State_Use)
        return false;

    void* weight = nullptr;
    auto& srcPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();
    auto* src = srcPtr->GetPtr();
    auto* dst = getChildEdgeAt(0)->getMemoryPtr()->GetPtr();
    auto& data_dims = srcPtr->getStaticDims();
    // M, K, N
    auto M = data_dims[0];
    auto K = weightDims[1];
    auto N = weightDims[0];
    if (data_dims.size() == 3) {
        M *= data_dims[1];
        assert(K == data_dims[2]);
    } else {
        assert(K == data_dims[1]);
    }
    auto work_amount = rnd_up(N, 32) / 32;
    float* bias = nullptr;
    if (withBiases) {
        bias = biasRnd.get();
    }
    auto thread_num = fcLLMs.size();
    size_t lda, ldb, ldc;
    lda = K * dnnl::memory::data_type_size(inputDataType);
    ldb = K * dnnl::memory::data_type_size(inputDataType);
    ldc = N * dnnl::memory::data_type_size(outputDataType);
    parallel_for(thread_num, [&](size_t tid) {
        size_t start {0}, end {0};
        dnnl::impl::balance211(work_amount, thread_num, tid, start, end);
        size_t n0 = start * 32;
        size_t n1 = std::min(end * 32, N);
        if (n0 >= N) return;

        llmdnn::fc_kernel_execute(fcLLMs[tid].get(), src, weight, dst, lda, ldb, ldc, M, N, K, n0, n1,
            dequant.get(), requant.get(), bias);
    });

    return true;
}

#endif

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
