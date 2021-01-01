#include "backend/kernel_compiler/cpu/kungfu_common.h"
#include "backend/kernel_compiler/cpu/kungfu_resize_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void KungFuResizeCPUKernel::InitKernel(const CNodePtr &kernel_node) { LOG_InitKernel("KungFuResizeCPUKernel"); }

bool KungFuResizeCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
  LOG_Kernel_Launch("KungFuResizeCPUKernel", inputs, workspace, outputs);
  uint32_t *p_new_size = reinterpret_cast<uint32_t *>(inputs.at(0)->addr);
  bool *pChanged = reinterpret_cast<bool *>(outputs.at(0)->addr);
  bool *pDetached = reinterpret_cast<bool *>(outputs.at(1)->addr);
  _kungfu_peer->ResizeCluster(*p_new_size, pChanged, pDetached);
  return true;
}

MS_REG_CPU_KERNEL(
  KungFuResize,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  KungFuResizeCPUKernel);
}  // namespace kernel
}  // namespace mindspore
