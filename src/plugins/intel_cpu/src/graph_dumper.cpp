// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_dumper.h"

#include "utils/debug_capabilities.h"
#include <ie_ngraph_utils.hpp>
#include "exec_graph_info.hpp"
#include "ie_common.h"
#include <dnnl_debug.h>
#include <ngraph/variant.hpp>
#include "ngraph/ngraph.hpp"
#include <ngraph/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#include <vector>
#include <string>
#include <memory>
#include <map>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

void serializeToCout(const Graph &graph);
void serializeToXML(const Graph &graph, const std::string& path);

namespace {

std::map<std::string, std::string> extract_node_metadata(const NodePtr &node) {
    std::map<std::string, std::string> serialization_info;

    if (node->getType() == Type::Input && node->isConstant()) {
        // We need to separate Input and Const layers
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = "Const";
    } else if (node->getType() == Type::Generic) {
        // Path to print actual name for extension layers
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = node->getTypeStr();
    } else {
        serialization_info[ExecGraphInfoSerialization::LAYER_TYPE] = NameFromType(node->getType());
    }

    // Original layers
    serialization_info[ExecGraphInfoSerialization::ORIGINAL_NAMES] = node->getOriginalLayers();

    // Implementation type name
    serialization_info[ExecGraphInfoSerialization::IMPL_TYPE] = node->getPrimitiveDescriptorType();

    std::string outputPrecisionsStr;
    if (!node->getChildEdges().empty()) {
        outputPrecisionsStr = node->getChildEdgeAt(0)->getMemory().getDesc().getPrecision().name();

        bool isAllEqual = true;
        for (size_t i = 1; i < node->getChildEdges().size(); i++) {
            if (node->getChildEdgeAt(i - 1)->getMemory().getDesc().getPrecision() != node->getChildEdgeAt(i)->getMemory().getDesc().getPrecision()) {
                isAllEqual = false;
                break;
            }
        }

        // If all output precisions are the same, we store the name only once
        if (!isAllEqual) {
            for (size_t i = 1; i < node->getChildEdges().size(); i++)
                outputPrecisionsStr += "," + std::string(node->getChildEdgeAt(i)->getMemory().getDesc().getPrecision().name());
        }
    } else {
        // Branch to correctly handle output nodes
        if (!node->getParentEdges().empty()) {
            outputPrecisionsStr = node->getParentEdgeAt(0)->getMemory().getDesc().getPrecision().name();
        }
    }
    serialization_info[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = outputPrecisionsStr;

    std::string outputLayoutsStr;
    auto outDescs = node->getSelectedPrimitiveDescriptor()->getConfig().outConfs;

    if (!outDescs.empty()) {
        outputLayoutsStr = outDescs[0].getMemDesc()->serializeFormat();

        bool isAllEqual = true;
        for (size_t i = 1; i < outDescs.size(); i++) {
            if (outDescs[i - 1].getMemDesc()->serializeFormat() != outDescs[i].getMemDesc()->serializeFormat()) {
                isAllEqual = false;
                break;
            }
        }

        // If all output layouts are the same, we store the name only once
        if (!isAllEqual) {
            for (size_t i = 1; i < outDescs.size(); i++) {
                outputLayoutsStr += "," + outDescs[i].getMemDesc()->serializeFormat();
            }
        }
    } else {
        outputLayoutsStr = dnnl::utils::fmt2str(dnnl::memory::format_tag::undef);
    }
    serialization_info[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = outputLayoutsStr;

    // Performance
    if (node->PerfCounter().avg() != 0) {
        serialization_info[ExecGraphInfoSerialization::PERF_COUNTER] = std::to_string(node->PerfCounter().avg());
    } else {
        serialization_info[ExecGraphInfoSerialization::PERF_COUNTER] = "not_executed";  // it means it was not calculated yet
    }

    serialization_info[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(node->getExecIndex());

    serialization_info[ExecGraphInfoSerialization::RUNTIME_PRECISION] = node->getRuntimePrecision().name();

    return serialization_info;
}

}  // namespace

std::shared_ptr<ngraph::Function> dump_graph_as_ie_ngraph_net(const Graph &graph) {
    std::map<NodePtr, std::shared_ptr<ngraph::Node> > node2layer;

    ngraph::ResultVector results;
    ngraph::ParameterVector params;
    ngraph::NodeVector to_hold;

    auto get_inputs = [&] (const NodePtr & node) {
        auto pr_edges = node->getParentEdges();
        ngraph::OutputVector inputs(pr_edges.size());

        for (int i = 0; i < pr_edges.size(); i++) {
            auto edge = node->getParentEdgeAt(i);
            int pr_port = edge->getInputNum();
            int ch_port = edge->getOutputNum();
            auto pr_node = edge->getParent();

            IE_ASSERT(node2layer.count(pr_node) == 1);
            auto pr = node2layer[pr_node];

            inputs[ch_port] = pr->output(pr_port);
        }

        return inputs;
    };

    auto create_ngraph_node = [&](const NodePtr &node) {
        bool is_input = false, is_output = false, should_be_hold = false;
        for (auto && kvp : graph.inputNodesMap) {
            if (kvp.second == node) {
                is_input = true;
                break;
            }
        }

        for (auto && kvp : graph.outputNodesMap) {
            if (kvp.second == node) {
                is_output = true;
                break;
            }
        }

        if (!is_output && node->getChildEdges().empty()) {
            // The node has no consumer and is not an output.
            // Should be hold in other irregular way.
            should_be_hold = true;
        }

        auto meta_data = extract_node_metadata(node);
        std::shared_ptr<ngraph::Node> return_node;
        if (is_input) {
            auto& desc = node->getChildEdgeAt(0)->getMemory().getDesc();
            auto param = std::make_shared<ngraph::op::Parameter>(details::convertPrecision(desc.getPrecision()), desc.getShape().toPartialShape());
            return_node = param;
            params.push_back(param);
        } else if (is_output) {
            results.emplace_back(std::make_shared<ngraph::op::Result>(get_inputs(node).back()));
            return_node = results.back();
        } else {
            return_node = std::make_shared<ExecGraphInfoSerialization::ExecutionNode>(
                get_inputs(node), node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size());

            for (size_t port = 0; port < return_node->get_output_size(); ++port) {
                auto& desc = node->getChildEdgeAt(port)->getMemory().getDesc();
                return_node->set_output_type(port, details::convertPrecision(desc.getPrecision()), desc.getShape().toPartialShape());
            }
        }

        if (should_be_hold) {
            to_hold.push_back(return_node);
        }

        for (auto && kvp : meta_data)
            return_node->get_rt_info()[kvp.first] = kvp.second;
        return_node->set_friendly_name(node->getName());

        return return_node;
    };

    ngraph::NodeVector nodes;
    nodes.reserve(graph.graphNodes.size());
    for (auto &node : graph.graphNodes) {  // important: graph.graphNodes are in topological order
        nodes.emplace_back(create_ngraph_node(node));
        node2layer[node] = nodes.back();
    }

    auto holder = results[0];
    for (auto &node : to_hold) {
        holder->add_control_dependency(node);
    }

    return std::make_shared<ngraph::Function>(results, params, graph._name);
}

#ifdef CPU_DEBUG_CAPS
void serialize(const Graph &graph) {
    const std::string& path = graph.getConfig().execGraphPath;

    if (path.empty())
        return;

    if (path == "cout")
        serializeToCout(graph);
    else if (!path.compare(path.size() - 4, 4, ".xml"))
        serializeToXML(graph, path);
    else
        IE_THROW() << "Unknown serialize format. Should be either 'cout' or '*.xml'. Got " << path;
}

void serializeToXML(const Graph &graph, const std::string& path) {
    if (path.empty())
        return;

    std::string binPath;
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(path,
                                               binPath,
                                               ov::pass::Serialize::Version::IR_V10);
    manager.run_passes(graph.dump());
}

struct str_separator {
    std::string sep;
    bool first;
    str_separator(const std::string & sep): sep(sep), first(true) {};
    void reset() { first = true; }
};

static std::ostream & operator<<(std::ostream & os, str_separator & sep) {
    if (!sep.first)
        os << sep.sep;
    else
        sep.first = false;
    return os;
};

void serializeToCout(const Graph &graph) {
    auto node_id = [](const NodePtr & node) {
        return std::string("t") + std::to_string(node->getExecIndex());
    };
    auto is_single_output_port = [](const NodePtr & node) {
        for(auto & e : node->getChildEdges()) {
            auto edge = e.lock();
            if (!edge) continue;
            if (edge->getInputNum() != 0)
                return false;
        }
        return true;
    };
    auto replace_all = [](std::string& inout, std::string what, std::string with) {
        std::size_t count{};
        for (std::string::size_type pos{};
            inout.npos != (pos = inout.find(what.data(), pos, what.length()));
            pos += with.length(), ++count) {
            inout.replace(pos, what.length(), with.data(), with.length());
        }
        return count;
    };

    str_separator comma(",");

    auto input_nodes = const_cast<Graph&>(graph).GetInputNodesMap();
    auto output_nodes = const_cast<Graph&>(graph).GetOutputNodesMap();

    comma.reset();
    for (auto it = output_nodes.begin(); it != output_nodes.end(); ++it) {
        std::cout << comma << it->first << "(" << node_id(it->second) << ")";
    }
    std::cout << " " << graph.GetName() << "(" << std::endl;
    for (auto it = input_nodes.begin(); it != input_nodes.end(); ++it) {
        std::cout << "\t" << it->first << "(" << node_id(it->second) << ")," << std::endl;
    }
    std::cout << ") {" << std::endl;

    for (const auto& node : graph.GetNodes()) {
        auto nodeDesc = node->getSelectedPrimitiveDescriptor();
        std::stringstream leftside;
        
        if (nodeDesc) {
            // output Desc is enough since input is always in consistent
            // with output.
            /*
            auto& inConfs = nodeDesc->getConfig().inConfs;
            if (!inConfs.empty()) {
                std::cout << " in:[";
                for (auto& c : inConfs) {
                    std::cout << c.getMemDesc()->getPrecision().name()
                            << c.getMemDesc()->
                            << "/" << c.getMemDesc()->serializeFormat()
                            << "; ";
                }
                std::cout << "]";
            }*/

            auto& outConfs = nodeDesc->getConfig().outConfs;
            if (!outConfs.empty()) {
                if (outConfs.size() > 1) leftside << "(";
                comma.reset();
                for (auto& c : outConfs) {
                    auto shape_str = c.getMemDesc()->getShape().toString();
                    replace_all(shape_str, "0 - ?", "?");
                    leftside << comma << c.getMemDesc()->getPrecision().name()
                              << "_" << c.getMemDesc()->serializeFormat()
                              << "_" << shape_str;
                }
                if (outConfs.size() > 1) leftside << ")";
            }
        }
        leftside << "  " << node_id(node) <<  " = ";
        std::cout << std::right << std::setw(40) << leftside.str();
        std::cout << std::left << node->getTypeStr();
        if (node->getAlgorithm() != Algorithm::Default)
            std::cout << "." << algToString(node->getAlgorithm());
        std::cout << " (";
        comma.reset();
        int id = 0;
        for (const auto & e : node->getParentEdges()) {
            auto edge = e.lock();
            if (!edge) continue;
            auto n = edge->getParent();
            
            std::cout << comma;
            if (edge->getOutputNum() != id)
                std::cout << "in" << edge->getOutputNum() << "=";

            std::cout << node_id(edge->getParent());

            if (!is_single_output_port(n))
                std::cout << "[" << edge->getInputNum() << "]";
            id ++;
        }

        if (node->getType() == intel_cpu::Type::Input && node->isConstant()) {
            auto & pmem = node->getChildEdgeAt(0)->getMemoryPtr();
            void * data = pmem->GetData();
            auto shape = pmem->getDesc().getShape().getDims();

            if (shape_size(shape) <= 8) {
                auto type = details::convertPrecision(pmem->getDesc().getPrecision());
                auto tensor = std::make_shared<ngraph::runtime::HostTensor>(type, shape, data);
                auto constop = std::make_shared<ngraph::op::Constant>(tensor);
                comma.reset();
                for (auto & v : constop->get_value_strings())
                    std::cout << comma << v;
            } else {
                std::cout << "...";
            }
        }

        std::cout << ")  ";
        std::cout << " " << node->getPrimitiveDescriptorType();

        auto split_str = [] (const std::string &str, const char delim) {
            std::vector<std::string> out;
            size_t start;
            size_t end = 0;
            while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
                end = str.find(delim, start);
                out.push_back(str.substr(start, end - start));
            }
            return out;
        };

        // last line(s): fused layers
        auto name = node->getName();
        auto vstr = split_str(node->getOriginalLayers(), ',');
        if (vstr.size() == 0) {
            std::cout << " " << name;
        } else {
            if ((vstr.size() == 1) && (vstr[0] == name)) {
                std::cout << " " << name;
            } else {
                if (name != vstr[0])
                    vstr.insert(vstr.begin(), name);
                for(auto& str : vstr)
                    std::cout << std::endl << std::setw(46) << std::right << " // " << str;
            }
        }
        std::cout << std::endl;
    }
    std::cout << "}" << std::endl;
}
#endif
}   // namespace intel_cpu
}   // namespace ov
