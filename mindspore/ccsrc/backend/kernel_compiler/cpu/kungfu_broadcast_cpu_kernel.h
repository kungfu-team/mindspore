#pragma once
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/kungfu_common.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {

template <typename T>
class KungFuBroadcastCPUKernel : public CPUKernel {
 public:
  KungFuBroadcastCPUKernel() {}
  ~KungFuBroadcastCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) { LOG_InitKernel("KungFuBroadcastCPUKernel"); };

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    LOG_Kernel_Launch("KungFuBroadcastCPUKernel", inputs, workspace, outputs);
    constexpr auto dtype = kungfu::type_encoder::value<T>();
    const auto px = inputs.at(0);
    const auto py = outputs.at(0);
    const auto count = px->size / sizeof(T);
    _kungfu_peer->Broadcast(px->addr, py->addr, count, dtype, "");
    return true;
  }
};
}  // namespace kernel
}  // namespace mindspore
