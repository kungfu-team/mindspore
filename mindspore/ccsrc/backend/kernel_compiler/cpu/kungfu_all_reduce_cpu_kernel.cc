#include "backend/kernel_compiler/cpu/kungfu_all_reduce_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

#include <iostream>
#include <memory>

#include <kungfu.h>

// int load_kungfu() {
//   //
// }

std::unique_ptr<kungfu::Peer> _kungfu_peer;

namespace mindspore {
namespace kernel {

void KungFuAllReduceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  std::cerr << "KungFuAllReduceCPUKernel::" << __func__ << " called" << std::endl;
  if (_kungfu_peer.get() == nullptr) {
    _kungfu_peer.reset(new kungfu::Peer);
  }
}

bool KungFuAllReduceCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  std::cerr << "KungFuAllReduceCPUKernel::" << __func__ << " called with " << inputs.size() << " inputs and "
            << outputs.size() << " outputs and " << workspace.size() << " workspaces" << std::endl;
  using T = float;
  constexpr auto dtype = kungfu::type_encoder::value<T>();
  constexpr auto op = KungFu_SUM;
  const int count = 1;  // TODO: get count
  _kungfu_peer->AllReduce(px->addr, py->addr, count, dtype, op, "");
  return true;
}

}  // namespace kernel
}  // namespace mindspore
