#include "backend/kernel_compiler/cpu/kungfu_common.h"
#include "backend/kernel_compiler/cpu/kungfu_broadcast_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

#include <memory>

namespace mindspore {
namespace kernel {
void KungFuBroadcastCPUKernel::InitKernel(const CNodePtr &kernel_node) { LOG_InitKernel("KungFuBroadcastCPUKernel"); }

bool KungFuBroadcastCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  LOG_Kernel_Launch("KungFuBroadcastCPUKernel", inputs, workspace, outputs);
  using T = float;
  constexpr auto dtype = kungfu::type_encoder::value<T>();
  const auto px = inputs.at(0);
  const auto py = outputs.at(0);
  const auto count = px->size / sizeof(T);
  _kungfu_peer->Broadcast(px->addr, py->addr, count, dtype, "");
  return true;
}
}  // namespace kernel
}  // namespace mindspore
