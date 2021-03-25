/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include "runtime/framework/actor/data_source_actor.h"
#include "runtime/framework/actor/loop_count_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/hardware/device_context.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;

enum class GraphExecutionStrategy {
  kPipeline,  // The actor running is triggered only by data.
  kStep       // The actor running need be triggered by control in addition.
};

// The actor set generated by graph transformer is the execution unit of actor runtime.
// It includes data source actor, kernel actor, loop count actor.
// The data source actor is used to obtain data and process them into device tensors,
// and then send them to kernel actor. The kernel actor is used to receive the device tensors to luanch kernel.
// Specifically notice the no input kernel actor, it means that this actor has no input device tensor, need be triggered
// externally. The loop count actor is used to receive the control of tail kernel actor to represent the end of one step
// and decide whether to loop execution by loop count.
struct ActorSet {
  std::vector<DataSourceActorPtr> data_source_actors_;
  std::vector<KernelActorPtr> kernel_actors_;
  // No input kernel actors need be triggered specifically.
  std::vector<KernelActorPtr> no_input_kernel_actors_;
  LoopCountActorPtr loop_count_actor_{nullptr};
};
using ActorSetPtr = std::shared_ptr<ActorSet>;

class GraphScheduler {
 public:
  static GraphScheduler &GetInstance() {
    static GraphScheduler instance;
    return instance;
  }

  // Transform graph to actor DAG, contains build and link.
  ActorSet *Transform(const KernelGraphPtr &graph, const DeviceContext *device_context,
                      const std::vector<tensor::TensorPtr> *input_tensors = nullptr,
                      GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

  // Schedule actors in the actor runtime. Single machine scheduling is supported currently, and distributed scheduling
  // will be supported in the future.
  void Schedule(const ActorSet *actor_set);

  // The processing entry of actors running.
  bool Run(const ActorSet *actor_set, GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

  // Fetch the actor set by kernel graph.
  ActorSet *Fetch(const KernelGraphPtr &graph) const;

 private:
  GraphScheduler() = default;
  ~GraphScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(GraphScheduler);

  // Transform the nodes of graph to actors.
  ActorSetPtr Build(const KernelGraphPtr &graph, const DeviceContext *device_context);
  // Link actors to DAG through the edge connection of graph and graph execution strategy.
  void Link(ActorSet *actor_set, const KernelGraphPtr &graph, GraphExecutionStrategy strategy);

  // The processing of actors build.
  std::vector<DataSourceActorPtr> BuildDataSourceActor(const KernelGraphPtr &graph);
  std::vector<KernelActorPtr> BuildKernelActor(const KernelGraphPtr &graph, const DeviceContext *device_context);
  std::vector<KernelActorPtr> BuildNoInputKernelActor(const KernelGraphPtr &graph);
  LoopCountActorPtr BuildLoopCountActor(const KernelGraphPtr &graph);

  // The processing of actors link.
  void LinkDataArrowForDeviceDSActor(DeviceQueueDataSourceActor *from_actor, KernelActor *to_actor,
                                     KernelWithIndex from_kernel_with_output_idx,
                                     KernelWithIndex to_to_kernel_with_input_idx);
  void LinkDataArrowForHostDSActor(HostQueueDataSourceActor *from_actor, KernelActor *to_actor,
                                   KernelWithIndex from_kernel_with_output_idx,
                                   KernelWithIndex to_kernel_with_input_idx);
  void LinkDataArrowForKernelActor(KernelActor *from_actor, KernelActor *to_actor,
                                   KernelWithIndex from_kernel_with_output_idx,
                                   KernelWithIndex to_kernel_with_input_idx);
  void LinkControlArrowForKernelActor(KernelActor *from_actor, LoopCountActor *to_actor, const KernelGraphPtr &graph,
                                      GraphExecutionStrategy strategy);
  void LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const KernelGraphPtr &graph);

  // Persist device tensors of graph's some nodes(such as weights and value nodes).
  void PersistDeviceTensor(const KernelGraphPtr &graph);

  std::unordered_map<KernelGraphPtr, ActorSetPtr> graph_to_actors_;
  std::unordered_map<KernelGraphPtr, HostTensorQueuePtr> graph_to_host_queue_;

  // The second element of pair represents the output index of kernel actor corresponding to the device tensor.
  std::unordered_map<DeviceTensorPtr, std::pair<KernelActorPtr, int>> device_address_to_actor_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_
