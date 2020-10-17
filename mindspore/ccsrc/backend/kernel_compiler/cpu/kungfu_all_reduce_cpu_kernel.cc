#include "backend/kernel_compiler/cpu/kungfu_common.h"
#include "backend/kernel_compiler/cpu/kungfu_all_reduce_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

#include <iostream>
#include <memory>

namespace mindspore {
namespace kernel {

void KungFuAllReduceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  std::cerr << "KungFuAllReduceCPUKernel::" << __func__ << " called" << std::endl;
  init_kungfu_once();
}

bool KungFuAllReduceCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  LOG_CALL("KungFuAllReduceCPUKernel", inputs, workspace, outputs);
  using T = float;
  constexpr auto dtype = kungfu::type_encoder::value<T>();
  constexpr auto op = KungFu_SUM;
  const auto px = inputs.at(0);
  const auto py = outputs.at(0);
  const auto count = px->size / sizeof(T);
  _kungfu_peer->AllReduce(px->addr, py->addr, count, dtype, op, "");
  return true;
}

}  // namespace kernel
}  // namespace mindspore
