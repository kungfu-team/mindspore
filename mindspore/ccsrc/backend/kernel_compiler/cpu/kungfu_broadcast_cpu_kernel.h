#pragma once
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {

// TODO: use template
class KungFuBroadcastCPUKernel : public CPUKernel {
 public:
  KungFuBroadcastCPUKernel() {}
  ~KungFuBroadcastCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
};

MS_REG_CPU_KERNEL(KungFuBroadcast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  KungFuBroadcastCPUKernel);

}  // namespace kernel
}  // namespace mindspore

