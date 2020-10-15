#include "backend/kernel_compiler/cpu/kungfu_all_reduce_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

#include <iostream>
#include <kungfu.h>

namespace mindspore {
namespace kernel {

void KungFuAllReduceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  std::cerr << "KungFuAllReduceCPUKernel::" << __func__ << " called" << std::endl;
}

bool KungFuAllReduceCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  std::cerr << "KungFuAllReduceCPUKernel::" << __func__ << " called with " << inputs.size() << " inputs and "
            << outputs.size() << " outputs and " << workspace.size() << " workspaces" << std::endl;
  const auto &px = inputs.at(0);
  const auto &py = outputs.at(0);

  std::cerr << px->size << std::endl;
  std::cerr << py->size << std::endl;

  using T = float;
  const T *x = reinterpret_cast<const T *>(px->addr);

  std::cerr << x[0] << std::endl;
  std::cerr << x[1] << std::endl;
  std::cerr << x[2] << std::endl;

  return true;
}

}  // namespace kernel
}  // namespace mindspore
