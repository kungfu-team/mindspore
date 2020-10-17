#pragma once
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {

class KungFuResizeCPUKernel : public CPUKernel {
 public:
  KungFuResizeCPUKernel() {}
  ~KungFuResizeCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
};

MS_REG_CPU_KERNEL(
  KungFuResize,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  KungFuResizeCPUKernel);

}  // namespace kernel
}  // namespace mindspore
