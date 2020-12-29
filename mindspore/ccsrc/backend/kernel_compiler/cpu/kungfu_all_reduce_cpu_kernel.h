#pragma once
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/kungfu_common.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
class KungFuAllReduceCPUKernel : public CPUKernel {
 public:
  KungFuAllReduceCPUKernel() {}
  ~KungFuAllReduceCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override { LOG_InitKernel("KungFuAllReduceCPUKernel"); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    LOG_Kernel_Launch("KungFuAllReduceCPUKernel", inputs, workspace, outputs);
    constexpr auto dtype = kungfu::type_encoder::value<T>();
    constexpr auto op = KungFu_SUM;
    const auto px = inputs.at(0);
    const auto py = outputs.at(0);
    const auto count = px->size / sizeof(T);
    _kungfu_peer->AllReduce(px->addr, py->addr, count, dtype, op, "");
    return true;
  }
};
}  // namespace kernel
}  // namespace mindspore
