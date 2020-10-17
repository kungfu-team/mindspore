#include "backend/kernel_compiler/cpu/kungfu_common.h"

std::unique_ptr<kungfu::Peer> _kungfu_peer;

namespace mindspore {
namespace kernel {

// TODO: mutex?
void init_kungfu_once() {
  if (_kungfu_peer.get() == nullptr) {
    _kungfu_peer.reset(new kungfu::Peer);
  }
}
}  // namespace kernel
}  // namespace mindspore
