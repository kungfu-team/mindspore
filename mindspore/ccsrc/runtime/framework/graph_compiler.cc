/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License"){}
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/framework/graph_compiler.h"
#include <numeric>
#include <map>
#include <utility>
#include "runtime/device/device_address.h"
#include "common/trans.h"
#include "utils/convert_utils.h"
#include "ir/tensor.h"
#include "backend/optimizer/common/helper.h"
#include "base/base_ref_utils.h"

namespace mindspore {
namespace runtime {
namespace {
// Whether device address of anf node is valid and device address type
// is consistent with device type, for example, device address type
// DeviceAddressType::kGPU should be used on GPU device
bool NodeDeviceAddressExist(const DeviceContext *device_context, const AnfNodePtr &kernel, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(device_context);
  if (AnfAlgo::OutputAddrExist(kernel, index)) {
    const auto &address = AnfAlgo::GetOutputAddr(kernel, index);
    MS_EXCEPTION_IF_NULL(address);
    return address->DeviceType() == device_context->GetDeviceAddressType();
  }
  return false;
}

void CreateParameterDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_inputs = graph->inputs();
  const std::vector<bool> &graph_valid_input = graph->valid_inputs();
  graph_inputs.insert(graph_inputs.end(), graph->child_graph_result().begin(), graph->child_graph_result().end());

  // Anf nodes which need create device address.
  std::vector<AnfNodePtr> nodes_list;
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    AnfNodePtr item = graph_inputs[i];
    MS_EXCEPTION_IF_NULL(item);
    if (i < graph_valid_input.size() && !graph_valid_input[i]) {
      continue;
    }

    if (AnfAlgo::CheckPrimitiveType(item, prim::kPrimMakeTuple)) {
      std::vector<AnfNodePtr> outs = AnfAlgo::GetAllOutput(item);
      for (const auto &out : outs) {
        MS_EXCEPTION_IF_NULL(out);
        if (!out->isa<Parameter>() || NodeDeviceAddressExist(device_context, out, 0)) {
          continue;
        }
        nodes_list.push_back(out);
      }
    }
    if (!item->isa<Parameter>() || NodeDeviceAddressExist(device_context, item, 0)) {
      continue;
    }
    nodes_list.push_back(item);
  }

  // Create device address for anf node in nodes_list
  for (const auto &item : nodes_list) {
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = AnfAlgo::GetOutputInferDataType(item, index);
      }

      size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      auto device_address = device_context->CreateDeviceAddress(nullptr, tensor_size,
                                                                AnfAlgo::GetOutputFormat(item, index), output_type_id);
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
}

void CreateDeviceAddressForTensorValue(const DeviceContext *device_context, const ValuePtr &node_value,
                                       size_t output_idx, const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::vector<TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);

  for (const auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (output_address != nullptr && output_address->DeviceType() == device_context->GetDeviceAddressType()) {
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), output_idx++,
                             value_node.get());
      continue;
    }

    size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(value_node, output_idx);
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown) {
      output_type_id = AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
    std::string output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);

    device::DeviceAddressPtr address =
      device_context->CreateDeviceAddress(nullptr, tensor_size, output_format, output_type_id);
    MS_EXCEPTION_IF_NULL(address);
    AnfAlgo::SetOutputAddr(address, output_idx++, value_node.get());
  }
}

void CreateValueNodeDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  for (const ValueNodePtr &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (NodeDeviceAddressExist(device_context, value_node, 0)) {
      continue;
    }

    const auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
      CreateDeviceAddressForTensorValue(device_context, node_value, 0, value_node);
    } else if (node_value->isa<StringImm>()) {
      auto value = GetValue<std::string>(node_value);
      size_t tensor_size = value.size();
      auto address = device_context->CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, kNumberTypeUInt8);
      MS_EXCEPTION_IF_NULL(address);

      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
    }
  }
}

void CreateKernelOutputDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      if (AnfAlgo::OutputAddrExist(kernel, i)) {
        continue;
      }

      std::string output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      auto device_address = device_context->CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type);
      AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
    }
  }
}

void CreateKernelWorkspaceDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      auto device_address = device_context->CreateDeviceAddress(nullptr, workspace_sizes[i], "", kTypeUnknown);
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}

void UpdateDeviceAddressForInplaceNode(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // Collect the inplace groups.
  std::map<uint32_t, std::vector<CNodePtr>> inplace_groups;
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    if (!AnfAlgo::IsInplaceNode(kernel, "inplace_algo")) {
      continue;
    }
    auto primitive = AnfAlgo::GetCNodePrimitive(kernel);
    MS_EXCEPTION_IF_NULL(primitive);
    auto inplace_group_attr = primitive->GetAttr("inplace_group");
    MS_EXCEPTION_IF_NULL(inplace_group_attr);
    auto group_id = GetValue<uint32_t>(inplace_group_attr);
    inplace_groups[group_id].emplace_back(kernel);
  }

  const size_t kMinInplaceGroupSize = 2;
  for (const auto &inplace_group : inplace_groups) {
    auto &group_nodes = inplace_group.second;
    if (group_nodes.size() < kMinInplaceGroupSize) {
      continue;
    }
    // Get the device address of the first node in the inplace group.
    auto node_primitive = AnfAlgo::GetCNodePrimitive(group_nodes[0]);
    MS_EXCEPTION_IF_NULL(node_primitive);
    auto output_index = GetValue<uint32_t>(node_primitive->GetAttr("inplace_output_index"));
    auto device_address = AnfAlgo::GetMutableOutputAddr(group_nodes[0], output_index, false);
    MS_EXCEPTION_IF_NULL(device_address);

    // Update the device address of other nodes using device address of the first node in the inplace group.
    for (size_t i = 1; i < group_nodes.size(); ++i) {
      auto &group_node = group_nodes[i];
      auto prim = AnfAlgo::GetCNodePrimitive(group_node);
      MS_EXCEPTION_IF_NULL(prim);
      auto index = GetValue<uint32_t>(prim->GetAttr("inplace_output_index"));
      AnfAlgo::SetOutputAddr(device_address, index, group_node.get());
      // Update the reference count of device address.
      device_address->IncreaseOriginalRefCount();
      device_address->ResetRefCount();
    }
  }
}

void SetSummaryNodesRefCount(const KernelGraph *graph) {
  if (!graph->summary_node_exist()) {
    return;
  }

  const std::map<std::string, std::pair<AnfNodePtr, int>> &summary_nodes = graph->summary_nodes();
  if (summary_nodes.empty()) {
    return;
  }

  for (const auto &item : summary_nodes) {
    const AnfNodePtr &node = item.second.first;
    size_t index = IntToSize(item.second.second);
    auto device_address = AnfAlgo::GetMutableOutputAddr(node, index, false);
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_original_ref_count(SIZE_MAX);
    device_address->ResetRefCount();
  }
}
}  // namespace

GraphId GraphCompiler::CompileGraph(const AnfNodePtrList &nodes, const AnfNodePtrList &outputs,
                                    const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(session_);
  // Generate kernel graph.
  KernelGraphPtr graph = session_->ConstructKernelGraph(nodes, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  return CompileGraphImpl(graph, device_context);
}

GraphId GraphCompiler::CompileGraphImpl(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);

  // Execute optimization pass.
  auto outputs_before_optimizer = AnfAlgo::GetAllOutputWithIndex(graph->output());
  device_context->OptimizeGraph(graph);
  auto outputs_after_optimizer = AnfAlgo::GetAllOutputWithIndex(graph->output());
  // Update the output map of kernel graph by modified output nodes.
  graph->UpdateGraphOutputMap(outputs_before_optimizer, outputs_after_optimizer);

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  device_context->CreateKernel(graph->execution_order());

  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph, device_context);

  graph->set_is_all_nop_node(opt::IsAllNopNode(graph.get()));

  MS_EXCEPTION_IF_NULL(session_);
  session_->InitAllBucket(graph, device_context);

  session_->SetSummaryNodes(graph.get());
  SetSummaryNodesRefCount(graph.get());

  return graph->graph_id();
}

GraphId GraphCompiler::CompileGraph(const session::OpRunInfo &op_run_info, const GraphInfo &graph_info,
                                    const std::vector<int64_t> *tensors_mask, std::vector<TensorPtr> *input_tensors,
                                    bool *single_op_cache_hit, const DeviceContext *device_context) {
  // Check if the graph cache exists.
  auto iter = run_op_graphs_.find(graph_info);
  if (iter != run_op_graphs_.end()) {
    const auto &graph = iter->second;
    MS_EXCEPTION_IF_NULL(graph);
    *single_op_cache_hit = true;
    return graph->graph_id();
  }
  *single_op_cache_hit = false;
  // Generate kernel graph.
  MS_EXCEPTION_IF_NULL(session_);
  KernelGraphPtr graph = session_->ConstructSingleOpGraph(op_run_info, *input_tensors, *tensors_mask);
  MS_EXCEPTION_IF_NULL(graph);

  MS_EXCEPTION_IF_NULL(device_context);
  device_context->OptimizeSingleOpGraph(graph);

  MS_EXCEPTION_IF_NULL(session_);
  session_->RunOpHideNopNode(graph);
  session_->RunOpRemoveNopNode(graph);

  // Generate 'KernelMod' for kernel in graph.
  device_context->CreateKernel(graph->execution_order());

  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph, device_context);

  graph->set_is_all_nop_node(opt::IsAllNopNode(graph.get()));
  run_op_graphs_[graph_info] = graph;
  return graph->graph_id();
}

KernelGraphPtr GraphCompiler::Fetch(GraphId graph_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetGraph(graph_id);
}

KernelGraphPtr GraphCompiler::Fetch(const GraphInfo &graph_info) const {
  auto iter = run_op_graphs_.find(graph_info);
  if (iter == run_op_graphs_.end()) {
    MS_LOG(ERROR) << "Can't find graph for: " << graph_info;
    return nullptr;
  }
  return iter->second;
}

void GraphCompiler::CreateDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  CreateParameterDeviceAddress(device_context, graph);
  CreateValueNodeDeviceAddress(device_context, graph);
  CreateKernelOutputDeviceAddress(device_context, graph);
  CreateKernelWorkspaceDeviceAddress(device_context, graph);
  UpdateDeviceAddressForInplaceNode(graph);
}

void GraphCompiler::GetParamAndOutputIndex(
  const KernelGraphPtr &graph, const std::vector<TensorPtr> &inputs, VectorRef *outputs,
  std::map<AnfNodePtr, size_t> *parameter_index,
  std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetParameterIndex(graph.get(), inputs, parameter_index);
  session_->CreateOutputPlaceholder(graph, inputs, outputs, output_indexes);
}

void GraphCompiler::GetSingleOpInputTensors(const CNodePtr &kernel,
                                            const std::map<KernelWithIndex, TensorPtr> &op_output,
                                            const std::map<AnfNodePtr, size_t> &parameter_index,
                                            const std::vector<TensorPtr> &graph_inputs,
                                            InputTensorInfo *input_tensor_info) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetOpInputTensors(kernel, op_output, parameter_index, graph_inputs, input_tensor_info);
}

void GraphCompiler::GetSingleOpRunInfoAndGraphInfo(const CNodePtr &kernel, const std::vector<TensorPtr> &input_tensors,
                                                   OpRunInfo *run_info, GraphInfo *graph_info) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetSingleOpRunInfo(kernel, run_info);
  *graph_info = session_->GetSingleOpGraphInfo(kernel, input_tensors);
}

void GraphCompiler::RecoverGraphOutput(
  const AnfNodePtr &kernel, const VectorRef &op_outputs,
  const std::map<KernelWithIndex, std::vector<std::vector<size_t>>> &output_indexes,
  std::map<KernelWithIndex, TensorPtr> *op_output_map, VectorRef *outputs,
  std::vector<TensorPtr> *runop_output_tensors) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(op_output_map);
  MS_EXCEPTION_IF_NULL(outputs);
  std::vector<TensorPtr> output_tensors = TransformVectorRefToMultiTensor(op_outputs);
  if (output_tensors.size() > op_outputs.size()) {
    MS_LOG(EXCEPTION) << "Op output contains tuple, node = " << kernel->DebugString();
  }
  size_t out_index = 0;
  for (const auto &output_tensor : output_tensors) {
    auto kernel_with_index = std::make_pair(kernel, out_index++);
    (*op_output_map)[kernel_with_index] = output_tensor;
    const auto &iter = output_indexes.find(kernel_with_index);
    if (iter == output_indexes.end()) {
      continue;
    }

    const std::vector<std::vector<size_t>> &multiple_ref_indexes = iter->second;
    for (const auto &ref_indexes : multiple_ref_indexes) {
      size_t n = 0;
      const VectorRef *cur_vector_ref = outputs;
      for (; n < ref_indexes.size() - 1; n += 1) {
        size_t index = ref_indexes.at(n);
        if (index >= cur_vector_ref->size()) {
          MS_LOG(EXCEPTION) << "Get invalid output ref index: " << index << ", size of vertor ref is "
                            << cur_vector_ref->size();
        }

        const BaseRef &base_ref = (*cur_vector_ref)[index];
        if (!utils::isa<VectorRef>(base_ref)) {
          MS_LOG(EXCEPTION) << "Get none VectorRef by ref index, index: " << index << "cur n: " << n;
        }
        cur_vector_ref = &utils::cast<VectorRef>(base_ref);
      }

      BaseRef &tensor_ref = (*const_cast<VectorRef *>(cur_vector_ref))[ref_indexes.at(n)];
      tensor_ref = output_tensor;
      runop_output_tensors->emplace_back(output_tensor);
    }
  }
}

void GraphCompiler::AddGradAddrToBucket(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &grad_tensor) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->AddGradAddrToBucket(graph_id, grad_tensor);
}

void GraphCompiler::ClearAllBucket(const GraphId &graph_id) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->ClearAllBucket(graph_id);
}

void GraphCompiler::RegisterSummaryCallBackFunc(const CallBackFunc &callback) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->RegisterSummaryCallBackFunc(callback);
}

void GraphCompiler::Summary(const std::vector<KernelGraphPtr> &graphs) const {
  MS_EXCEPTION_IF_NULL(session_);
  for (const auto &graph : graphs) {
    session_->Summary(graph.get());
  }
}
}  // namespace runtime
}  // namespace mindspore