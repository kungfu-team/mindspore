#include "backend/kernel_compiler/cpu/kungfu_all_reduce_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

#include <iostream>

namespace mindspore {
namespace kernel {

void KungFuAllReduceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  std::cerr << "KungFuAllReduceCPUKernel::" << __func__ << " called" << std::endl;
}

bool KungFuAllReduceCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  std::cerr << "KungFuAllReduceCPUKernel::" << __func__ << " called" << std::endl;
  return true;
}

}  // namespace kernel
}  // namespace mindspore
